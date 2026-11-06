"""E9 (BRIDGE-Hybrid) context blocks + regenerated E1 / E5 comparison contexts on the CLEANED graphs.

E9 block = whole-building schema skeleton (bridge.retrieve.schema_skeleton on the cleaned v1-format graph,
           i.e. exactly the hybrid E5 block)
         + GNN query candidates: class paths from a beam search over instance edges of up to 7 hops
           (expansion restricted to the top-N entity nodes by the GNN node head; typed nodes only),
           length-diverse selection, compiled to SPARQL (projection from the answer head) and executed
           on the cleaned ABox + TBox (row count + sample rows)
         + an ATTRIBUTE MENU for every entity variable of each candidate: the literal-valued paths
           actually present on the entities bound in the candidate's result (direct literal predicates,
           IRI values such as qudt:hasUnit, and 2-hop paths through untyped / blank nodes such as
           ref:hasExternalReference/ref:hasTimeseriesId), each with coverage and an example value, ordered
           by the GNN answer-head probability of the corresponding attribute nodes.
         Total block <= ~5k tokens (candidates are dropped from the tail to fit).

python -m bridge.hybrid.context_e9 contexts --dataset devlit [--configs E1 E5 E9] [--ckpt runs/v2/ckpt_vm/v2main.pt]
python -m bridge.hybrid.context_e9 coverage --dataset devlit     # offline: gold path / attributes in candidates?
Outputs: runs/hybrid/contexts/{dataset}/contexts_E{1,5,9}.json (+ _meta), runs/hybrid/coverage_{dataset}.json
"""
import argparse
import collections
import json
import time

import numpy as np
import pyoxigraph as ox
import torch

from bridge import kg, qsets
from bridge.hybrid import common as H
from bridge.labels import run_with_timeout
from bridge.retrieve import (TripleIndex, bridge_scores, compile_path_query, embed, knn_context, lift_key,
                             load_model, n_tokens, schema_skeleton, var_name)
from bridge.v2 import common as C2

DEFAULT_CKPT = H.ROOT / "runs" / "v2" / "ckpt_vm" / "v2main.pt"
TOTAL_BUDGET = 5000
QTIMEOUT = 10


# ----------------------------------------------------------------------------- per-building resources
class BRes:
    """Cleaned v2 InfGraph + cleaned store + adjacency for attribute menus."""

    def __init__(self, b, dev):
        self.b = b
        self.ig = H.inf_graph(b, dev, "v2")
        name = ("bq_" + b) if b in H.BQ_BUILDINGS else b
        self.store = kg.make_store(H.clean_nt(name), kg.tbox_path(H.stratum(b)))
        self.idx = {n: i for i, n in enumerate(self.ig.g["node_ids"])}
        self.lits = self.ig._lits
        self.obj = collections.defaultdict(list)
        self.typed = set()
        for t in ox.parse(path=str(H.clean_nt(name)), format=ox.RdfFormat.N_TRIPLES):
            if isinstance(t.object, ox.NamedNode):
                if t.predicate.value == kg.RDF_TYPE:
                    self.typed.add(t.subject.value)
                else:
                    self.obj[t.subject.value].append((t.predicate.value, t.object.value))


# ----------------------------------------------------------------------------- beam search (long)
@torch.no_grad()
def beam_long(model, ig, out, seeds, allowed, width=16, max_hops=7, fanout=400):
    pt = ig.g["primary_type"]
    ei = ig.g["edge_index"]
    beams = [((s,), (), 0.0) for s in seeds]
    found = []
    for _ in range(max_hops):
        cand_e, cand_b = [], []
        for bi, (nodes, edges, sc) in enumerate(beams):
            for e in ig.out_edges.get(nodes[-1], [])[:fanout]:
                v = int(ei[1, e])
                if v in nodes or not allowed[v] or int(pt[v]) < 0:
                    continue
                cand_e.append(e)
                cand_b.append(bi)
        if not cand_e:
            break
        _, lg = model.edge_repr(out, 0, torch.tensor(cand_e, device=ig.dev), ig.t)
        scored = sorted(((beams[b][2] + l, b, e) for l, b, e in zip(lg.tolist(), cand_b, cand_e)), reverse=True)
        seen, nb = set(), []
        for sc, b, e in scored[: width * 6]:
            nodes, edges, _ = beams[b]
            cand = (nodes + (int(ei[1, e]),), edges + (e,), sc)
            key = lift_key(ig, cand[0], cand[1])
            if key in seen:
                continue
            seen.add(key)
            nb.append(cand)
            if len(nb) >= width:
                break
        beams = nb
        found += beams
    return found


def select_paths(agg, n_top=5, per_len=2, max_total=16):
    """Best paths overall + the best per_len paths of every length >= 3 (length diversity)."""
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    sel = [k for k, _ in ranked[:n_top]]
    by_len = collections.defaultdict(list)
    for k, _ in ranked:
        by_len[len(k) // 2].append(k)
    for L in sorted(by_len):
        if L >= 3:
            sel += [k for k in by_len[L][:per_len] if k not in sel]
    sel = sorted(set(sel), key=lambda k: -agg[k])
    return sel[:max_total]


def path_var_names(ig, key):
    names, used = [], collections.Counter()
    for i in range(0, len(key), 2):
        base = var_name(ig.g["node_ids"][key[i]])[1:] or "x"
        used[base] += 1
        names.append(base if used[base] == 1 else f"{base}{used[base]}")
    return names


def run_rows(store, sparql, cap=2000):
    rows, err = run_with_timeout(store, sparql + f"\nLIMIT {cap}", QTIMEOUT)
    if err or rows is None:
        return None, []
    names = sparql.split("SELECT DISTINCT ", 1)[1].split(" WHERE", 1)[0].replace("?", "").split()
    return names, rows


# ----------------------------------------------------------------------------- attribute menus
def attribute_menu(res, ans_prob, entities, top=6):
    """[(path tuple, kind, coverage, example, score)] of literal-valued paths on these entities."""
    ig = res.ig
    stats = collections.defaultdict(lambda: {"n": 0, "ex": None, "score": 0.0, "kind": "lit"})
    for s in entities:
        seen = set()
        for p, v in res.lits.get(s, ()):
            a = C2.attr_id(s, p, v)
            st = stats[(p,)]
            if (p,) not in seen:
                st["n"] += 1; seen.add((p,))
            st["ex"] = st["ex"] or v
            if a in res.idx:
                st["score"] = max(st["score"], float(ans_prob[res.idx[a]]))
        for p, o in res.obj.get(s, ()):
            if o in res.typed:
                continue
            if not o.startswith("urn:bnode:") and not res.lits.get(o) and not res.obj.get(o):
                st = stats[(p,)]                         # IRI value (unit:DEG_F, quantitykind:..., ...)
                st["kind"] = "iri"
                if (p,) not in seen:
                    st["n"] += 1; seen.add((p,))
                st["ex"] = st["ex"] or ig.pfx.c(o)
                if o in res.idx:
                    st["score"] = max(st["score"], float(ans_prob[res.idx[o]]))
                continue
            for p2, v in res.lits.get(o, ()):          # 2 hops through an untyped / blank node
                key = (p, p2)
                st = stats[key]
                if key not in seen:
                    st["n"] += 1; seen.add(key)
                st["ex"] = st["ex"] or v
                a = C2.attr_id(o, p2, v)
                if a in res.idx:
                    st["score"] = max(st["score"], float(ans_prob[res.idx[a]]))
    out = [(k, d["kind"], d["n"], d["ex"], d["score"]) for k, d in stats.items()]
    out.sort(key=lambda x: (-x[4], -x[2], len(x[0])))
    return out[:top]


def menu_line(ig, var, n_ent, menu):
    parts = []
    for path, kind, n, ex, sc in menu:
        ps = "/".join(ig.pfx.c(p) for p in path)
        exs = ex if kind == "iri" else '"' + str(ex)[:40].replace("\n", " ") + '"'
        parts.append(f"{ps} e.g. {exs} ({n}/{n_ent})")
    return f"  ?{var}: " + (" | ".join(parts) if parts else "(no literal attributes)")


# ----------------------------------------------------------------------------- E9
@torch.no_grad()
def e9_candidates(model, res, question, max_hops=7, n_cand=8, n_allowed=3000, width=16):
    ig = res.ig
    t0 = time.time()
    q = embed([question]).to(ig.dev)
    out = bridge_scores(model, ig, q)
    nl = out["node_logit"][0]
    n_inst = ig.g["n_inst"]
    ent = torch.zeros_like(nl, dtype=torch.bool)
    ent[:n_inst] = True
    ent &= ~ig.is_attr.to(nl.device)
    ent &= (ig.g["primary_type"] >= 0).to(nl.device)
    score = nl.masked_fill(~ent, -1e9)
    k = min(n_allowed, int(ent.sum()))
    allowed = torch.zeros_like(ent)
    allowed[score.topk(k).indices] = True
    allowed = allowed.cpu().tolist()
    seeds = score.topk(min(10, k)).indices.tolist()
    beams = beam_long(model, ig, out, seeds, allowed, width=width, max_hops=max_hops)
    agg = {}
    for nodes, edges, sc in beams:
        key = lift_key(ig, nodes, edges)
        if any(key[i] < 0 for i in range(0, len(key), 2)):
            continue
        agg[key] = max(agg.get(key, -1e9), sc / max(1, len(edges)) ** 0.5)
    sel = select_paths(agg)
    ans_prob = torch.sigmoid(out["ans_logit"][0]).cpu()
    pt = ig.g["primary_type"]
    ent_ans = ans_prob.masked_fill(~ent.cpu(), 0)
    ans_classes = {int(pt[i]) for i in ent_ans.topk(min(30, ent_ans.numel())).indices.tolist() if int(pt[i]) >= 0}
    pfx = dict(ig.pfx.items)
    cands, seen = [], set()
    for key in sel:
        if tuple(reversed(key)) in seen:
            continue
        seen.add(key)
        sparql = compile_path_query(ig, key, ans_classes, pfx)
        names_p, rows_p = run_rows(res.store, sparql)
        if not rows_p:
            continue
        sig = (len(rows_p), frozenset(tuple(sorted(kg.term_str(r[n]) for n in names_p if r[n] is not None))
                                      for r in rows_p[:50]))
        if sig in seen:
            continue
        seen.add(sig)
        vnames = path_var_names(ig, key)
        all_q = sparql.replace("SELECT DISTINCT " + sparql.split("SELECT DISTINCT ", 1)[1].split(" WHERE", 1)[0],
                               "SELECT DISTINCT " + " ".join("?" + v for v in vnames), 1)
        _, rows_a = run_rows(res.store, all_q)
        menus = []
        for v in vnames:
            ents = list(dict.fromkeys(kg.term_str(r[v]) for r in rows_a if r[v] is not None))[:200]
            menus.append((v, len(ents), attribute_menu(res, ans_prob, ents)))
        sample = [[ig.pfx.c(kg.term_str(r[n]))[:60] if r[n] is not None else "" for n in names_p] for r in rows_p[:3]]
        cands.append({"key": list(key), "hops": len(key) // 2, "sparql": sparql, "n_rows": len(rows_p),
                      "cols": names_p, "sample": sample, "menus": menus, "score": agg[key]})
        if len(cands) >= n_cand:
            break
    return cands, time.time() - t0


def e9_block(skeleton, cands, ig, budget=TOTAL_BUDGET):
    head = ["## Candidate query skeletons proposed by the graph retriever (already executed on this graph)",
            "Ranked by the retriever's relevance to the question (paths of up to 7 hops). Each candidate lists an"
            " attribute menu: literal-valued properties actually present on the entities of each variable"
            " (direct, or via an untyped / external-reference node), with coverage and an example value,"
            " best-scored first. Use, combine or extend candidates; add the attribute columns the question asks for."]
    used = n_tokens(skeleton) + n_tokens("\n".join(head)) + 10
    parts, shown = list(head), []
    for i, c in enumerate(cands, 1):
        body = c["sparql"].split("SELECT", 1)[1]
        txt = (f"### Candidate {i} ({c['hops']} hops): {c['n_rows']} rows; columns {', '.join('?' + x for x in c['cols'])}\n"
               f"SELECT{body}\nsample: " + " | ".join(", ".join(r) for r in c["sample"])
               + "\nattribute menu:\n" + "\n".join(menu_line(ig, v, n, m) for v, n, m in c["menus"]))
        if used + n_tokens(txt) > budget and shown:
            # try a compact version (menus of projected columns only) before giving up
            txt = (txt.split("\nattribute menu:")[0] + "\nattribute menu:\n"
                   + "\n".join(menu_line(ig, v, n, m[:3]) for v, n, m in c["menus"] if v in c["cols"]))
            if used + n_tokens(txt) > budget:
                break
        parts.append(txt)
        used += n_tokens(txt)
        shown.append(c)
    block = skeleton + "\n\n" + "\n".join(parts)
    return block, shown


# ----------------------------------------------------------------------------- drivers
def by_building(dataset):
    d = collections.defaultdict(list)
    for q in qsets.questions(dataset):
        d[q["building"]].append(q)
    return d


def cmd_contexts(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdir = H.CTX / a.dataset
    cdir.mkdir(parents=True, exist_ok=True)
    model = None
    if "E9" in a.configs:
        model, mcfg = load_model(a.ckpt, dev)
    outs = {c: {} for c in a.configs}
    for b, items in by_building(a.dataset).items():
        t0 = time.time()
        ig1 = H.inf_graph(b, dev, "v1") if ("E1" in a.configs or "E5" in a.configs or "E9" in a.configs) else None
        # NB: not bridge.retrieve.e5_context - its cache is keyed by id(ig), which Python may reuse for the
        # next building's graph once the previous one is garbage-collected (stale skeleton)
        skel = schema_skeleton(ig1) if ig1 is not None else None
        if "E5" in a.configs:
            for q in items:
                outs["E5"][q["qkey"]] = {"block": skel, "tokens": n_tokens(skel)}
        if "E1" in a.configs:
            idx = TripleIndex(ig1)
            for q in items:
                blk, info = knn_context(idx, ig1, q["text"])
                outs["E1"][q["qkey"]] = {"block": blk, "tokens": info["tokens"], "n_triples": info["n_triples"]}
        if "E9" in a.configs:
            res = BRes(b, dev)
            for q in items:
                cands, dt = e9_candidates(model, res, q["text"], max_hops=a.max_hops, width=a.width,
                                          n_allowed=a.n_allowed)
                blk, shown = e9_block(skel, cands, res.ig)
                outs["E9"][q["qkey"]] = {"block": blk, "tokens": n_tokens(blk), "skeleton_tokens": n_tokens(skel),
                                         "latency_s": round(dt, 2), "n_candidates": len(shown),
                                         "hops": [c["hops"] for c in shown],
                                         "candidates": [c["sparql"] for c in shown],
                                         "candidates_all": [c["sparql"] for c in cands],
                                         "menus": [{v: [[list(p), k, n] for p, k, n, _, _ in m] for v, _, m in c["menus"]}
                                                   for c in shown]}
        msg = " ".join(f"{c} {np.mean([outs[c][q['qkey']]['tokens'] for q in items]):.0f}tok" for c in a.configs)
        if "E9" in a.configs:
            msg += f" E9 {np.mean([outs['E9'][q['qkey']]['latency_s'] for q in items]):.1f}s/q"
        print(f"{b}: {len(items)} q, {msg} ({time.time() - t0:.0f}s)", flush=True)
    for c, d in outs.items():
        (cdir / f"contexts_{c}{a.suffix if c == 'E9' else ''}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    meta_f = cdir / "contexts_meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    for c in a.configs:
        meta[c + (a.suffix if c == "E9" else "")] = {"ckpt": str(a.ckpt) if c == "E9" else None, "graphs": "hybrid cleaned",
                   "max_hops": a.max_hops, "width": a.width, "n_allowed": a.n_allowed,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta_f.write_text(json.dumps(meta, indent=1))
    for c, d in outs.items():
        tk = [v["tokens"] for v in d.values()]
        print(f"{a.dataset} {c}: {len(d)} contexts, tokens mean {np.mean(tk):.0f} median {np.median(tk):.0f} max {max(tk)}")


# ----------------------------------------------------------------------------- offline coverage check
def gold_structure(ctx, sparql):
    """Gold entity steps {(cls_s, p, cls_o)} (most frequent primary types over rows) and literal attribute
    tails {(cls_owner, path)} of a gold query."""
    from bridge.labels import parse_patterns, select_star
    from rdflib import Variable
    edges, tpats, other, proj = parse_patterns(sparql)
    rows, err = run_with_timeout(ctx.store, select_star(sparql), 60)
    if err or not rows:
        return None
    prim = ctx.prim

    def val(sol, t):
        if isinstance(t, Variable):
            try:
                return sol[str(t)]
            except (KeyError, ValueError):
                return None
        return ox.NamedNode(str(t))

    step_c = collections.defaultdict(collections.Counter)
    attr_c = collections.defaultdict(collections.Counter)
    by_subj = collections.defaultdict(list)
    for s, p, o in edges:
        by_subj[str(s)].append((p, o))
    for sol in rows[:500]:
        for k, (s, p, o) in enumerate(edges):
            sv, ov = val(sol, s), val(sol, o)
            if not isinstance(sv, ox.NamedNode):
                continue
            cs = prim.get(sv.value)
            if isinstance(ov, ox.NamedNode) and prim.get(ov.value) and cs:
                step_c[k][(cs, str(p), prim[ov.value])] += 1
            elif cs and isinstance(ov, ox.Literal):
                attr_c[k][(cs, (str(p),))] += 1
            elif cs and isinstance(ov, ox.NamedNode) and not prim.get(ov.value):
                # 2-hop tail through an untyped node, or an IRI value
                tails = [(p2, o2) for p2, o2 in by_subj.get(str(o), [])]
                if not tails:
                    attr_c[k][(cs, (str(p),))] += 1
                for p2, o2 in tails:
                    v2 = val(sol, o2)
                    if isinstance(v2, ox.Literal) or (isinstance(v2, ox.NamedNode) and not prim.get(v2.value)):
                        attr_c[k][(cs, (str(p), str(p2)))] += 1
    steps = {c.most_common(1)[0][0] for c in step_c.values() if c}
    attrs = {c.most_common(1)[0][0] for c in attr_c.values() if c}
    return {"steps": steps, "attrs": attrs}


def cand_structure(sparql, ns_map=None):
    from bridge.labels import parse_patterns
    from rdflib import Variable
    try:
        edges, tpats, _, _ = parse_patterns(sparql)
    except Exception:  # noqa: BLE001
        return set(), {}
    cls = collections.defaultdict(set)
    for s, c in tpats:
        cls[str(s)].add(str(c))
    steps = set()
    for s, p, o in edges:
        if isinstance(s, Variable) and isinstance(o, Variable):
            for a in cls.get(str(s), ()):
                for b in cls.get(str(o), ()):
                    steps.add((a, str(p), b))
    return steps, cls


def e9_menu_attrs(ctx_rec, sparql_list):
    """(class, path) pairs offered in the E9 attribute menus of the shown candidates."""
    out = set()
    for sparql, menus in zip(sparql_list, ctx_rec.get("menus", [])):
        _, cls = cand_structure(sparql)
        for v, entries in menus.items():
            for c in cls.get(v, ()):
                for path, _, _ in entries:
                    out.add((c, tuple(path)))
    return out


def coverage_of(gold, cand_sparqls, menu_attrs=None):
    G = gold["steps"]
    cs = [cand_structure(s)[0] for s in cand_sparqls]
    union = set().union(*cs) if cs else set()
    r = {"has_steps": bool(G),
         "single": bool(G) and any(G <= c for c in cs),
         "union": bool(G) and G <= union,
         "step_recall": (len(G & union) / len(G)) if G else None}
    if menu_attrs is not None:
        A = gold["attrs"]
        r["attr_recall"] = (len(A & menu_attrs) / len(A)) if A else None
        single_full = False
        for s, c in zip(cand_sparqls, cs):
            if G <= c:
                classes = {x for st in c for x in (st[0], st[2])}
                if all(a[0] in classes and a in menu_attrs for a in A):
                    single_full = True
        r["single_full"] = bool(G) and single_full
    return r


def old_e7_candidates(dataset, dev):
    """E7 candidates of the earlier attempt: stored for dev223 (runs/dev223/contexts_E7.json); recomputed
    for devlit with the same v1 model / v1 graphs / store construction as eval_buildingqa contexts-v7."""
    f = H.ROOT / "runs" / dataset / "contexts_E7.json"
    if f.exists():
        return {k: v["candidates"] for k, v in json.loads(f.read_text(encoding="utf-8")).items()}
    from bridge.eval_buildingqa import dev_graph
    from bridge.retrieve import gnn_query_candidates
    model, _ = load_model(H.ROOT / "runs" / "ckpt" / "main.pt", dev)
    out = {}
    for b, items in by_building(dataset).items():
        ig = dev_graph(b, dev)
        store = kg.make_store(items[0]["ttl"], items[0]["tbox"], abox_format=ox.RdfFormat.TURTLE)
        for q in items:
            out[q["qkey"]] = [c["sparql"] for c in gnn_query_candidates(model, ig, q["text"], store)]
    return out


def cmd_coverage(a):
    from bridge.hybrid.synth_long import hybrid_ctx
    assert a.dataset != "buildingqa", "offline coverage uses gold SPARQL: dev sets only"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e9 = json.loads((H.CTX / a.dataset / f"contexts_E9{a.suffix}.json").read_text(encoding="utf-8"))
    e7 = old_e7_candidates(a.dataset, dev)
    items = json.loads((H.ROOT / "runs" / "dev" / f"{a.dataset.replace('dev', 'dev_set', 1)}.json").read_text(encoding="utf-8"))
    ctxs, per = {}, []
    for it in items:
        b = it["building"]
        if b not in ctxs:
            ctxs = {b: hybrid_ctx(b)}
        gold = gold_structure(ctxs[b], it["gold_sparql"])
        if gold is None:
            continue
        rec = e9[it["qkey"]]
        r9 = coverage_of(gold, rec["candidates"], e9_menu_attrs(rec, rec["candidates"]))
        r9all = coverage_of(gold, rec.get("candidates_all", rec["candidates"]))
        r7 = coverage_of(gold, e7.get(it["qkey"], []))
        per.append({"qkey": it["qkey"], "stratum": it["stratum"], "n_steps": len(gold["steps"]),
                    "n_attrs": len(gold["attrs"]), "n_hops": it.get("n_hops"), "E9": r9, "E9_all": r9all, "E7": r7,
                    "E9_tokens": rec["tokens"], "E9_hops": rec["hops"]})

    def agg(rows, key, field):
        v = [r[key][field] for r in rows if r[key].get(field) is not None and (field != "single" or r[key]["has_steps"])]
        return round(float(np.mean(v)), 3) if v else None

    summary = {}
    for st in sorted({r["stratum"] for r in per}) + ["ALL"]:
        rows = [r for r in per if st == "ALL" or r["stratum"] == st]
        rs = [r for r in rows if r["E9"]["has_steps"]]
        summary[st] = {"n": len(rows), "n_with_entity_steps": len(rs),
                       "E7_single": agg(rs, "E7", "single"), "E7_union": agg(rs, "E7", "union"),
                       "E7_step_recall": agg(rs, "E7", "step_recall"),
                       "E9_single": agg(rs, "E9", "single"), "E9_union": agg(rs, "E9", "union"),
                       "E9_step_recall": agg(rs, "E9", "step_recall"),
                       "E9_all_single": agg(rs, "E9_all", "single"),
                       "E9_attr_recall": agg(rows, "E9", "attr_recall"),
                       "E9_single_full": agg(rs, "E9", "single_full"),
                       "E9_tokens_mean": round(float(np.mean([r["E9_tokens"] for r in rows])), 0)}
    long = [r for r in per if r["n_steps"] >= 3 and r["E9"]["has_steps"]]
    summary["gold>=3_entity_steps"] = {"n": len(long), "E7_single": agg(long, "E7", "single"),
                                      "E9_single": agg(long, "E9", "single"),
                                      "E7_step_recall": agg(long, "E7", "step_recall"),
                                      "E9_step_recall": agg(long, "E9", "step_recall")}
    out = {"dataset": a.dataset, "summary": summary, "per_question": per,
           "definitions": {"single": "all gold entity steps (class,pred,class) contained in ONE candidate",
                           "union": "all gold entity steps covered by the union of shown candidates",
                           "step_recall": "share of gold entity steps covered by the union",
                           "attr_recall": "share of gold literal/IRI attribute tails (owner class, path) offered in E9 menus",
                           "single_full": "one candidate covers all entity steps AND its menus offer all gold attributes",
                           "E9_all": "all executed E9 candidates before the token-budget cut"}}
    (H.H / f"coverage_{a.dataset}{a.suffix}.json").write_text(json.dumps(out, indent=1))
    for k, v in summary.items():
        print(a.dataset, k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["contexts", "coverage"])
    ap.add_argument("--dataset", required=True, choices=["buildingqa", "dev", "dev223", "devlit"])
    ap.add_argument("--configs", nargs="*", default=["E1", "E5", "E9"])
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--max-hops", type=int, default=7)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--n-allowed", type=int, default=3000)
    ap.add_argument("--suffix", default="", help="E9 variant suffix, e.g. _h4 -> contexts_E9_h4.json (ablations)")
    a = ap.parse_args()
    {"contexts": cmd_contexts, "coverage": cmd_coverage}[a.cmd](a)
