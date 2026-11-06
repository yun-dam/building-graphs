"""Step 7: BRIDGE inference -> class-level paths -> verbalized context block (<= ~2k tokens).

Also provides the text-kNN context (E1) with the same token budget.
"""
import collections
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from bridge import kg
from bridge.data import schema_graph
from bridge.graph_build import build, embed
from bridge.model import BRIDGE

ROOT = kg.ROOT
BQ_DIR = ROOT / "data" / "BuildingQA" / "eval_buildings"
GBQ = ROOT / "runs" / "graphs_bq"
TOKEN_BUDGET = 2000
CHARS_PER_TOKEN = 4
GT_KEYS = ["x", "edge_index", "edge_type", "rel_x", "rel_graph", "rel_graph_type", "rel_ids"]
STD = {
    "brick": kg.BRICK, "s223": kg.S223, "ref": "https://brickschema.org/schema/Brick/ref#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "qudt": "http://qudt.org/schema/qudt/", "unit": "http://qudt.org/vocab/unit/",
    "quantitykind": "http://qudt.org/vocab/quantitykind/", "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def n_tokens(s):
    return len(s) // CHARS_PER_TOKEN


# ----------------------------------------------------------------------------- graphs
class Prefixes:
    def __init__(self, ttl=None):
        self.map = dict(STD)
        if ttl:
            with open(ttl, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*@?prefix\s+(\w*):\s*<([^>]+)>", line, re.I)
                    if m and m.group(2) not in self.map.values():
                        self.map[m.group(1) or "bldg"] = m.group(2)
        self.items = sorted(self.map.items(), key=lambda kv: -len(kv[1]))

    def c(self, iri):
        if iri.startswith("urn:bnode:"):
            return "[]"
        for p, ns in self.items:
            if iri.startswith(ns) and re.fullmatch(r"[\w\-\.]+", iri[len(ns):] or "x"):
                return f"{p}:{iri[len(ns):]}"
        return f"<{iri}>"


class InfGraph:
    """A building graph prepared for inference."""

    def __init__(self, g, prefixes, dev):
        self.g = g
        self.pfx = prefixes
        self.dev = dev
        self.t = {k: g[k].to(dev) for k in GT_KEYS if k in g}
        self.t["n_fwd_edges"] = g["n_fwd_edges"]
        names = g["rel_names"]
        if "rel_ids" not in g:
            import zlib
            self.t["rel_ids"] = torch.tensor([zlib.crc32(n.encode()) % 4096 for n in names], device=dev)
        self.steps, self.adj = schema_graph(g)
        self.skip_rel = {i for i, n in enumerate(names) if n.lstrip("^") in (kg.RDF_TYPE, kg.SUBCLASS)}
        ei, et = g["edge_index"], g["edge_type"]
        self.out_edges = collections.defaultdict(list)
        for e in range(ei.size(1)):
            if int(et[e]) not in self.skip_rel:
                self.out_edges[int(ei[0, e])].append(e)
        self._lits = None

    def rel_str(self, r):
        n = self.g["rel_names"][r]
        return ("^" + self.pfx.c(n[1:])) if n.startswith("^") else self.pfx.c(n)

    def node_str(self, i):
        return self.pfx.c(self.g["node_ids"][i])

    def cls_str(self, i):
        return self.pfx.c(self.g["node_ids"][i]) if i >= 0 else "?untyped"


def bq_graph(building, dev):
    """BuildingQA / anonymized eval graph, built read-only from eval_buildings/*.ttl (cached)."""
    GBQ.mkdir(parents=True, exist_ok=True)
    f = GBQ / f"{building}.pt"
    ttl = BQ_DIR / f"{building}.ttl"
    stratum = "s223" if building in ("b59", "bldg_223p_anon") else "brick"
    if not f.exists():
        g = build("bq_" + building, kg.nt_cache("bq_" + building, ttl), stratum)
        torch.save(g, f)
    g = torch.load(f, weights_only=False)
    return InfGraph(g, Prefixes(ttl), dev)


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    c = ck["config"]
    m = BRIDGE(d=c["d"], rel_layers=c["rel_layers"], ent_layers=c["ent_layers"], dropout=c["dropout"],
               hard_seeding=c["hard_seeding"], relation_graph=not c["no_relation_graph"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, c


# ----------------------------------------------------------------------------- BRIDGE
@torch.no_grad()
def bridge_scores(model, ig, q):
    """Run the model; graphs > 100k edges use the 4-hop subgraph around the top soft seeds."""
    t = ig.t
    E = t["n_fwd_edges"]
    if E > 100_000:
        cos = q @ t["x"].float().t()
        seeds = cos.topk(model.top_seeds, dim=1).indices[0]
        subset, sub_ei, _, emask = k_hop_subgraph(seeds, 4, t["edge_index"], relabel_nodes=True,
                                                  num_nodes=t["x"].size(0))
        sub = dict(t)
        sub["x"], sub["edge_index"], sub["edge_type"] = t["x"][subset], sub_ei, t["edge_type"][emask]
        sub["n_fwd_edges"] = 0  # not used by forward
        out = model(sub, q)
        N = t["x"].size(0)
        full = {}
        for k in ("node_logit", "ans_logit"):
            v = torch.full((1, N), -1e4, device=q.device)
            v[0, subset] = out[k][0]
            full[k] = v
        h = torch.zeros(1, N, out["h"].size(-1), device=q.device)
        h[0, subset] = out["h"][0]
        full["h"], full["w"] = h, out["w"]
        return full
    return model(t, q)


@torch.no_grad()
def beam_paths(model, ig, out, seeds, width=10, max_hops=4):
    """Beam search over instance edges from seed nodes, scored by summed edge logits."""
    beams = [((s,), (), 0.0) for s in seeds]
    found = []
    for _ in range(max_hops):
        cand_e, cand_b = [], []
        for bi, (nodes, edges, sc) in enumerate(beams):
            for e in ig.out_edges.get(nodes[-1], [])[:400]:
                v = int(ig.g["edge_index"][1, e])
                if v in nodes:
                    continue
                cand_e.append(e)
                cand_b.append(bi)
        if not cand_e:
            break
        e_t = torch.tensor(cand_e, device=ig.dev)
        _, lg = model.edge_repr(out, 0, e_t, ig.t)
        lg = lg.tolist()
        scored = sorted(((beams[b][2] + l, b, e) for l, b, e in zip(lg, cand_b, cand_e)), reverse=True)
        new = []
        for sc, b, e in scored[: width * 4]:
            nodes, edges, _ = beams[b]
            new.append((nodes + (int(ig.g["edge_index"][1, e]),), edges + (e,), sc))
        # keep diversity at class level
        seen, beams = set(), []
        for nb in new:
            key = lift_key(ig, nb[0], nb[1])
            if key in seen:
                continue
            seen.add(key)
            beams.append(nb)
            if len(beams) >= width:
                break
        found += beams
    return found


def lift_key(ig, nodes, edges):
    pt = ig.g["primary_type"]
    et = ig.g["edge_type"]
    return tuple(x for i, n in enumerate(nodes) for x in ((int(pt[n]),) + ((int(et[edges[i]]),) if i < len(edges) else ())))


def path_count(ig, key, cap=100000):
    """Number of instance paths realizing a class-level path (DP over schema step edge sets)."""
    ei = ig.g["edge_index"]
    cur = None
    for i in range(1, len(key), 2):
        es = ig.steps.get((key[i - 1], key[i], key[i + 1]), [])
        nxt = collections.Counter()
        for e in es:
            u, v = int(ei[0, e]), int(ei[1, e])
            c = 1 if cur is None else cur.get(u, 0)
            if c:
                nxt[v] += c
        cur = nxt
        if not cur:
            return 0
    return min(sum(cur.values()), cap)


def class_path_str(ig, key):
    s = ig.cls_str(key[0])
    for i in range(1, len(key), 2):
        r = ig.g["rel_names"][key[i]]
        if r.startswith("^"):
            s += f" <--{ig.pfx.c(r[1:])}-- {ig.cls_str(key[i + 1])}"
        else:
            s += f" --{ig.pfx.c(r)}--> {ig.cls_str(key[i + 1])}"
    return s


def var_name(cls_iri):
    ln = kg.local_name(cls_iri)
    return "?" + re.sub(r"[^a-z0-9]+", "_", ln.lower()).strip("_")[:30]


@torch.no_grad()
def bridge_context(model, ig, question, top_nodes=50, budget=TOKEN_BUDGET):
    t0 = time.time()
    q = embed([question]).to(ig.dev)
    out = bridge_scores(model, ig, q)
    nl = out["node_logit"][0]
    n_inst = ig.g["n_inst"]
    top = nl.topk(min(top_nodes, nl.numel())).indices.tolist()
    seeds = [i for i in top if i < n_inst][:10]
    beams = beam_paths(model, ig, out, seeds)
    agg = {}
    for nodes, edges, sc in beams:
        key = lift_key(ig, nodes, edges)
        if any(key[i] < 0 for i in range(0, len(key), 2)):
            continue
        a = agg.setdefault(key, {"score": sc / max(1, len(edges)) ** 0.5, "ex": []})
        a["score"] = max(a["score"], sc / max(1, len(edges)) ** 0.5)
        if len(a["ex"]) < 3:
            a["ex"].append(nodes)
    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["score"])
    # answer-variable hint: classes of top answer-head nodes
    al = out["ans_logit"][0][:n_inst]
    pt = ig.g["primary_type"]
    hint_cls = []
    for i in al.topk(min(30, al.numel())).indices.tolist():
        c = int(pt[i])
        if c >= 0 and c not in hint_cls:
            hint_cls.append(c)
        if len(hint_cls) >= 4:
            break
    head = "## Relevant graph structure (retrieved for this question)\nPaths (class-level, with counts):\n"
    lines, uris = [], set()
    used = n_tokens(head) + 150
    for key, a in ranked:
        cnt = path_count(ig, key)
        if cnt == 0:
            continue
        ex = " ; ".join(" -> ".join(ig.node_str(n) for n in nodes) for nodes in a["ex"][:2])
        line = f"{len(lines) + 1}. {class_path_str(ig, key)}   [{cnt} paths; e.g. {ex}]"
        if used + n_tokens(line) > budget:
            break
        lines.append(line)
        used += n_tokens(line)
        for i in range(0, len(key), 2):
            uris.add(ig.cls_str(key[i]))
        for i in range(1, len(key), 2):
            uris.add(ig.pfx.c(ig.g["rel_names"][key[i]].lstrip("^")))
    hints = ", ".join(f"{var_name(ig.g['node_ids'][c])} ({ig.cls_str(c)})" for c in hint_cls)
    for c in hint_cls:
        uris.add(ig.cls_str(c))
    block = head + "\n".join(lines) + f"\nLikely answer variables: {hints}\nExact URIs present in this graph: " + ", ".join(sorted(uris))
    return block, {"latency_s": time.time() - t0, "n_paths": len(lines), "tokens": n_tokens(block),
                   "class_paths": [list(k) for k, _ in ranked[: len(lines)]],
                   "top_nodes": top, "out": out}


# ----------------------------------------------------------------------------- text-kNN
class TripleIndex:
    """Verbalized ABox triples (forward edges, incl. rdf:type) with frozen MiniLM embeddings."""

    def __init__(self, ig):
        g = ig.g
        E = g["n_fwd_edges"]
        ei, et = g["edge_index"][:, :E], g["edge_type"][:E]
        texts, lines = [], []
        for e in range(E):
            s, o, r = int(ei[0, e]), int(ei[1, e]), int(et[e])
            texts.append(f"{g['texts'][s]} {g['rel_texts'][r]} {g['texts'][o]}"[:400])
            lines.append(f"{ig.node_str(s)} {ig.rel_str(r)} {ig.node_str(o)} .")
        self.texts, self.lines = texts, lines
        self.emb = embed(texts).to(ig.dev)
        self.ei = ei

    def rank(self, question):
        q = embed([question]).to(self.emb.device)
        return (self.emb @ q[0]).argsort(descending=True)


def knn_context(index, ig, question, budget=TOKEN_BUDGET):
    t0 = time.time()
    order = index.rank(question).tolist()
    head = "## Relevant graph triples (retrieved for this question by text similarity)\n"
    lines, uris = [], set()
    used = n_tokens(head) + 100
    for e in order:
        ln = index.lines[e]
        if used + n_tokens(ln) > budget:
            break
        lines.append(ln)
        used += n_tokens(ln) + 1
        r = int(ig.g["edge_type"][e])
        uris.add(ig.pfx.c(ig.g["rel_names"][r].lstrip("^")))
        o = int(index.ei[1, e])
        if ig.g["is_class"][o]:
            uris.add(ig.node_str(o))
    block = head + "\n".join(lines) + "\nExact URIs present in this graph: " + ", ".join(sorted(uris))
    return block, {"latency_s": time.time() - t0, "n_triples": len(lines), "tokens": n_tokens(block),
                   "edges": order[: len(lines)]}


# ----------------------------------------------------------------------------- BRIDGE v3 (E3)
def literal_attrs(ig):
    """node IRI -> [(predicate IRI, literal value)] from the same N-Triples file the graph was built from."""
    if ig._lits is None:
        import pyoxigraph as ox
        name = ig.g["building"] + ("@compiled" if ig.g.get("graph_variant") == "compiled" else "")
        lits = collections.defaultdict(list)
        for t in ox.parse(path=str(kg.NTDIR / f"{name}.nt"), format=ox.RdfFormat.N_TRIPLES):
            if isinstance(t.object, ox.Literal):
                lits[t.subject.value].append((t.predicate.value, t.object.value))
        ig._lits = lits
    return ig._lits


def _lit_str(ig, pairs, k=3, maxlen=60):
    return " ; ".join(f'{ig.pfx.c(p)} "{v[:maxlen]}"' for p, v in pairs[:k])


def node_rich(ig, i):
    """Compact node name; blank nodes are inlined with their literal attributes."""
    iri = ig.g["node_ids"][i]
    if iri.startswith("urn:bnode:"):
        lits = literal_attrs(ig).get(iri, [])
        return f"[ {_lit_str(ig, lits, k=2)} ]" if lits else "[]"
    return ig.node_str(i)


@torch.no_grad()
def bridge_context_v3(model, ig, question, top_nodes=50, budget=TOKEN_BUDGET, n_paths=8):
    """E3: class-level paths + highest-scored instance triples on those paths + answer-head
    candidates + literal attributes of the retrieved instances (same trained model, no retraining)."""
    t0 = time.time()
    q = embed([question]).to(ig.dev)
    out = bridge_scores(model, ig, q)
    nl = out["node_logit"][0]
    n_inst = ig.g["n_inst"]
    top = nl.topk(min(top_nodes, nl.numel())).indices.tolist()
    seeds = [i for i in top if i < n_inst][:10]
    beams = beam_paths(model, ig, out, seeds)
    agg = {}
    for nodes, edges, sc in beams:
        key = lift_key(ig, nodes, edges)
        if any(key[i] < 0 for i in range(0, len(key), 2)):
            continue
        s = sc / max(1, len(edges)) ** 0.5
        a = agg.setdefault(key, {"score": s, "edges": set()})
        a["score"] = max(a["score"], s)
        a["edges"] |= set(edges)
    ranked = [(k, a) for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["score"]) if path_count(ig, k) > 0][:n_paths]
    ei, et = ig.g["edge_index"], ig.g["edge_type"]
    E = ig.g["n_fwd_edges"]
    uris = set()
    # A. class-level paths (short)
    head = "## Relevant graph structure (retrieved for this question)\nPaths (class-level, with counts):\n"
    lines_a, used = [], n_tokens(head) + 120
    for key, a in ranked:
        line = f"{len(lines_a) + 1}. {class_path_str(ig, key)}   [{path_count(ig, key)} paths]"
        if used + n_tokens(line) > 0.3 * budget:
            break
        lines_a.append(line)
        used += n_tokens(line)
        uris |= {ig.cls_str(key[i]) for i in range(0, len(key), 2)}
        uris |= {ig.pfx.c(ig.g["rel_names"][key[i]].lstrip("^")) for i in range(1, len(key), 2)}
    # B. instance triples on those paths, ranked by the edge head
    cand = set()
    for key, a in ranked[: len(lines_a)]:
        cand |= a["edges"]
        for i in range(1, len(key), 2):
            cand |= set(ig.steps.get((key[i - 1], key[i], key[i + 1]), [])[:300])
    lines_b, shown_nodes, seen = [], [], set()
    if cand:
        ce = torch.tensor(sorted(cand), device=ig.dev)
        _, lg = model.edge_repr(out, 0, ce, ig.t)
        for e in ce[lg.argsort(descending=True)].tolist():
            f = e - E if e >= E else e
            if f in seen:
                continue
            seen.add(f)
            s, o, r = int(ei[0, f]), int(ei[1, f]), int(et[f])
            line = f"{node_rich(ig, s)} {ig.rel_str(r)} {node_rich(ig, o)} ."
            if used + n_tokens(line) > 0.75 * budget:
                break
            lines_b.append(line)
            used += n_tokens(line) + 1
            shown_nodes += [s, o]
    # C. answer-variable candidates (answer head)
    al = out["ans_logit"][0][:n_inst]
    pt = ig.g["primary_type"]
    groups = collections.OrderedDict()
    for i in al.topk(min(40, al.numel())).indices.tolist():
        c = int(pt[i])
        if c >= 0 and (c in groups or len(groups) < 4) and len(groups.get(c, [])) < 5:
            groups.setdefault(c, []).append(i)
    lines_c = []
    for c, ns in groups.items():
        line = f"{var_name(ig.g['node_ids'][c])} ({ig.cls_str(c)}): " + ", ".join(node_rich(ig, n) for n in ns)
        lines_c.append(line)
        used += n_tokens(line)
        uris.add(ig.cls_str(c))
    # D. literal attributes of retrieved instances
    lits = literal_attrs(ig)
    lines_d, done = [], set()
    for n in [n for ns in groups.values() for n in ns] + shown_nodes:
        iri = ig.g["node_ids"][n]
        if n in done or iri.startswith("urn:bnode:") or not lits.get(iri):
            continue
        done.add(n)
        line = f"{ig.node_str(n)}  {_lit_str(ig, lits[iri])}"
        if used + n_tokens(line) > budget - 60:
            break
        lines_d.append(line)
        used += n_tokens(line)
        uris |= {ig.pfx.c(p) for p, _ in lits[iri][:3]}
    block = (head + "\n".join(lines_a)
             + "\nInstance triples on these paths (highest-scored first):\n" + "\n".join(lines_b)
             + "\nLikely answer variables and top candidate instances:\n" + "\n".join(lines_c)
             + ("\nLiteral attributes of retrieved instances:\n" + "\n".join(lines_d) if lines_d else "")
             + "\nExact URIs present in this graph: " + ", ".join(sorted(uris)))
    return block, {"latency_s": time.time() - t0, "n_paths": len(lines_a), "n_triples": len(lines_b),
                   "tokens": n_tokens(block), "class_paths": [list(k) for k, _ in ranked[: len(lines_a)]]}


# ----------------------------------------------------------------------------- E4: global skeleton + E3
def schema_skeleton(ig, budget=3500):
    """Question-independent class-level summary of the WHOLE building graph: every
    (class --predicate--> class) with instance-edge counts, plus literal attributes per class
    (and per untyped node reached via a predicate, e.g. external-reference blank nodes)."""
    g = ig.g
    ei, et = g["edge_index"], g["edge_type"]
    E = g["n_fwd_edges"]
    pt = g["primary_type"]
    rels = g["rel_names"]
    skip = {kg.RDF_TYPE, kg.SUBCLASS}
    edges = collections.Counter()
    untyped_vals = collections.defaultdict(collections.Counter)
    parent = {}
    for e in range(E):
        r = int(et[e])
        if rels[r] in skip:
            continue
        s, o = int(ei[0, e]), int(ei[1, e])
        a, b = int(pt[s]), int(pt[o])
        if a >= 0:
            edges[(a, r, b)] += 1
            if b < 0:
                parent.setdefault(o, (a, r))
                untyped_vals[(a, r)][o] += 1
    lits = literal_attrs(ig)
    idx = {n: i for i, n in enumerate(g["node_ids"])}
    attrs = collections.defaultdict(collections.Counter)
    for iri, pairs in lits.items():
        i = idx.get(iri)
        if i is None:
            continue
        c = int(pt[i])
        key = ig.cls_str(c) if c >= 0 else (
            f"[{ig.cls_str(parent[i][0])} --{ig.rel_str(parent[i][1])}--> untyped node]" if i in parent else None)
        if key is None:
            continue
        for p, _ in pairs:
            attrs[key][ig.pfx.c(p)] += 1
    head = ("## Whole-building schema skeleton (class-level, all relations with instance counts)\n")
    out, used = [head], n_tokens(head)
    for (a, r, b), n in edges.most_common():
        if b >= 0:
            tgt = ig.cls_str(b)
        else:
            vals = untyped_vals[(a, r)]
            named = [ig.node_str(o) for o, _ in vals.most_common(6) if not g["node_ids"][o].startswith("urn:bnode:")]
            tgt = ("{" + ", ".join(named) + (", ..." if len(vals) > 6 else "") + "}") if named else "untyped node"
        line = f"{ig.cls_str(a)} --{ig.rel_str(r)}--> {tgt}  ({n})"
        if used + n_tokens(line) > 0.65 * budget:
            out.append("... (more relations omitted)")
            break
        out.append(line)
        used += n_tokens(line)
    out.append("Literal attributes by class (predicate: count):")
    for key, cnt in sorted(attrs.items(), key=lambda kv: -sum(kv[1].values())):
        line = f"{key}: " + ", ".join(f"{p} ({n})" for p, n in cnt.most_common(8))
        if used + n_tokens(line) > budget:
            out.append("... (more classes omitted)")
            break
        out.append(line)
        used += n_tokens(line)
    return "\n".join(out)


def e4_context(model, ig, question, skeleton_cache={}):
    key = id(ig)
    if key not in skeleton_cache:
        skeleton_cache[key] = schema_skeleton(ig)
    blk, info = bridge_context_v3(model, ig, question)
    full = skeleton_cache[key] + "\n\n" + blk
    info = dict(info, tokens=n_tokens(full), skeleton_tokens=n_tokens(skeleton_cache[key]))
    return full, info


def e5_context(ig, skeleton_cache={}):
    """E5 ablation: whole-building schema skeleton only (no learned retrieval)."""
    key = id(ig)
    if key not in skeleton_cache:
        skeleton_cache[key] = schema_skeleton(ig)
    return skeleton_cache[key], {"tokens": n_tokens(skeleton_cache[key])}


def oracle_context(ig, label_rec, budget=TOKEN_BUDGET):
    """E6 diagnostic (dev only): the GOLD evidence of the query (Y_E instance triples, gold class
    skeleton P_ast, gold answer nodes) verbalized in the E3 format. Upper bound for retrieval."""
    idx = {n: i for i, n in enumerate(ig.g["node_ids"])}
    pfx = ig.pfx.c
    head = "## Relevant graph structure (retrieved for this question)\nPaths (class-level):\n"
    lines_a = [f"{k + 1}. {pfx(s) if s else '?'} --{pfx(p)}--> {pfx(o) if o else '?'}"
               for k, (s, p, o) in enumerate(label_rec.get("P_ast") or [])]
    used = n_tokens(head) + sum(n_tokens(x) for x in lines_a) + 100
    lines_b = []
    for s, p, o in label_rec["Y_E"]:
        if p == kg.RDF_TYPE:
            continue
        sn = node_rich(ig, idx[s]) if s in idx else pfx(s)
        on = node_rich(ig, idx[o]) if o in idx else pfx(o)
        line = f"{sn} {pfx(p)} {on} ."
        if used + n_tokens(line) > 0.8 * budget:
            break
        lines_b.append(line)
        used += n_tokens(line) + 1
    ans = [node_rich(ig, idx[n]) for n in label_rec["Y_ans"][:15] if n in idx]
    block = (head + "\n".join(lines_a) + "\nInstance triples on these paths (highest-scored first):\n" + "\n".join(lines_b)
             + "\nLikely answer variables and top candidate instances:\n" + ", ".join(ans))
    return block, {"tokens": n_tokens(block)}


# ----------------------------------------------------------------------------- E7: GNN-compiled executable queries
def compile_path_query(ig, key, ans_classes, prefixes):
    """Class-level path key (C0, r1, C1, ..., Ck) -> SPARQL with one variable per class.
    Projection = path variables whose class is predicted by the answer head (else both ends)."""
    names, used = [], collections.Counter()
    for i in range(0, len(key), 2):
        base = var_name(ig.g["node_ids"][key[i]])[1:] or "x"
        used[base] += 1
        names.append(base if used[base] == 1 else f"{base}{used[base]}")
    pats = [f"?{names[0]} a {ig.cls_str(key[0])} ."]
    for j, i in enumerate(range(1, len(key), 2)):
        r = ig.g["rel_names"][key[i]]
        a, b = names[j], names[j + 1]
        pats.append(f"?{b} {ig.pfx.c(r[1:])} ?{a} ." if r.startswith("^") else f"?{a} {ig.pfx.c(r)} ?{b} .")
        pats.append(f"?{b} a {ig.cls_str(key[i + 1])} .")
    proj = [names[k] for k in range(len(names)) if key[2 * k] in ans_classes]
    if not proj:
        proj = [names[0], names[-1]] if len(names) > 1 else [names[0]]
    pre = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in prefixes.items())
    return f"{pre}\nSELECT DISTINCT {' '.join('?' + v for v in dict.fromkeys(proj))} WHERE {{\n  " + "\n  ".join(pats) + "\n}"


@torch.no_grad()
def gnn_query_candidates(model, ig, question, store, n_cand=5, n_paths=12):
    """Top GNN class paths compiled to SPARQL and executed (row count + 3 sample rows each)."""
    q = embed([question]).to(ig.dev)
    out = bridge_scores(model, ig, q)
    nl = out["node_logit"][0]
    n_inst = ig.g["n_inst"]
    top = nl.topk(min(50, nl.numel())).indices.tolist()
    beams = beam_paths(model, ig, out, [i for i in top if i < n_inst][:10])
    agg = {}
    for nodes, edges, sc in beams:
        k = lift_key(ig, nodes, edges)
        if any(k[i] < 0 for i in range(0, len(k), 2)):
            continue
        agg[k] = max(agg.get(k, -1e9), sc / max(1, len(edges)) ** 0.5)
    ranked = [k for k, _ in sorted(agg.items(), key=lambda kv: -kv[1])][:n_paths]
    al = out["ans_logit"][0][:n_inst]
    pt = ig.g["primary_type"]
    ans_classes = {int(pt[i]) for i in al.topk(min(30, al.numel())).indices.tolist() if int(pt[i]) >= 0}
    used_pfx = {}
    for p, ns in ig.pfx.items:
        used_pfx[p] = ns
    cands, seen = [], set()
    for k in ranked:
        # canonical direction: skip a path that is the reverse of one already compiled
        rev = tuple(reversed(k))
        if rev in seen:
            continue
        seen.add(k)
        sparql = compile_path_query(ig, k, ans_classes, {p: ns for p, ns in used_pfx.items()
                                                         if f"{p}:" in class_path_str(ig, k) or p in ("rdf", "rdfs")})
        try:
            sols = store.query(sparql, use_default_graph_as_union=True)
            names = [v.value for v in sols.variables]
            rows = []
            for i, sol in enumerate(sols):
                if i >= 2000:
                    break
                rows.append([ig.pfx.c(kg.term_str(sol[n])) if sol[n] is not None else "" for n in names])
        except Exception as e:  # noqa: BLE001
            continue
        if rows:
            cands.append({"sparql": sparql, "n_rows": len(rows), "cols": names, "sample": rows[:3]})
        if len(cands) >= n_cand:
            break
    return cands


def e7_context(model, ig, question, store, skeleton_cache={}):
    """E7: whole-building skeleton + GNN-proposed executable query skeletons with their results."""
    key = id(ig)
    if key not in skeleton_cache:
        skeleton_cache[key] = schema_skeleton(ig)
    cands = gnn_query_candidates(model, ig, question, store)
    parts = ["## Candidate query skeletons proposed by the graph retriever (already executed on this graph)",
             "They are ranked by the retriever's relevance to the question. Use, combine or extend them;"
             " add the columns the question asks for."]
    for i, c in enumerate(cands, 1):
        body = c["sparql"].split("SELECT", 1)[1]
        parts.append(f"### Candidate {i}: {c['n_rows']} rows; columns {', '.join('?' + x for x in c['cols'])}\n"
                     f"SELECT{body}\nsample: " + " | ".join(", ".join(r) for r in c["sample"]))
    block = skeleton_cache[key] + "\n\n" + "\n".join(parts)
    return block, {"tokens": n_tokens(block), "n_candidates": len(cands), "candidates": cands}
