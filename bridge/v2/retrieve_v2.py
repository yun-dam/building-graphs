"""E8: v2 GNN (with attribute nodes) proposes executable query skeletons INCLUDING the literal
attribute columns its answer head selects. Context = same whole-building skeleton as E5/E7 +
these candidates (executed, with sample rows)."""
import collections
import zlib
from pathlib import Path

import torch

from bridge import kg
from bridge.retrieve import (InfGraph, Prefixes, beam_paths, bridge_scores, class_path_str, compile_path_query,
                             embed, lift_key, n_tokens, var_name)

ROOT = kg.ROOT
V2G = ROOT / "runs" / "v2" / "graphs"
V2GBQ = ROOT / "runs" / "v2" / "graphs_bq"
BQ_DIR = ROOT / "data" / "BuildingQA" / "eval_buildings"


def _finish(g, ttl, dev):
    g["rel_ids"] = torch.tensor([zlib.crc32(n.encode()) % 4096 for n in g["rel_names"]], dtype=torch.long)
    ig = InfGraph(g, Prefixes(ttl), dev)
    ig.is_attr = g["is_attr"]
    return ig


def v2_dev_graph(b, dev):
    g = torch.load(V2G / f"{b}.pt", weights_only=False)
    return _finish(g, ROOT / kg.manifest()[b]["path"], dev)


def v2_bq_graph(b, dev):
    """BuildingQA graph in v2 format (attribute nodes), built read-only from eval_buildings/*.ttl."""
    from bridge.v2.graph_build_v2 import build
    V2GBQ.mkdir(parents=True, exist_ok=True)
    f = V2GBQ / f"{b}.pt"
    ttl = BQ_DIR / f"{b}.ttl"
    if not f.exists():
        stratum = "s223" if b in ("b59", "bldg_223p_anon") else "brick"
        torch.save(build("bq_" + b, kg.nt_cache("bq_" + b, ttl), stratum), f)
    return _finish(torch.load(f, weights_only=False), ttl, dev)


def attribute_columns(ig, ans_prob, thr=0.5, top=4):
    """(class id, predicate path) -> best answer-head probability over attribute nodes reachable from
    instances of that class directly (s -p-> attr) or via one untyped node (s -q-> u -p-> attr)."""
    g = ig.g
    ei, et = g["edge_index"], g["edge_type"]
    E = g["n_fwd_edges"]
    pt = g["primary_type"]
    is_attr = ig.is_attr
    rels = g["rel_names"]
    src, dst, rt = ei[0, :E].tolist(), ei[1, :E].tolist(), et[:E].tolist()
    parent = {}
    for s, o, r in zip(src, dst, rt):
        if not is_attr[o] and int(pt[o]) < 0 and int(pt[s]) >= 0 and not rels[r].startswith(("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",)):
            parent.setdefault(o, (int(pt[s]), r))
    best = {}
    for s, o, r in zip(src, dst, rt):
        if not is_attr[o]:
            continue
        p = float(ans_prob[o])
        c = int(pt[s])
        if c >= 0:
            key = (c, (r,))
        elif s in parent:
            c, q = parent[s]
            key = (c, (q, r))
        else:
            continue
        if p > best.get(key, 0.0):
            best[key] = p
    ranked = sorted(((k, v) for k, v in best.items() if v >= thr), key=lambda kv: -kv[1])
    return ranked[: top * 3]


def augment_query(ig, base_sparql, key, attrs, max_cols=3):
    """Add attribute columns for path classes to a compiled path query."""
    path_cls = [key[i] for i in range(0, len(key), 2)]
    names, used = [], collections.Counter()
    for c in path_cls:
        b = var_name(ig.g["node_ids"][c])[1:] or "x"
        used[b] += 1
        names.append(b if used[b] == 1 else f"{b}{used[b]}")
    add_pats, add_vars = [], []
    for (c, preds), _ in attrs:
        if c not in path_cls or len(add_vars) >= max_cols:
            continue
        v = names[path_cls.index(c)]
        loc = ig.pfx.c(ig.g["rel_names"][preds[-1]]).split(":")[-1].strip("<>").split("/")[-1].split("#")[-1]
        av = f"{v}_{''.join(ch for ch in loc if ch.isalnum())[:20]}"
        if av in add_vars:
            continue
        if len(preds) == 1:
            add_pats.append(f"?{v} {ig.rel_str(preds[0])} ?{av} .")
        else:
            add_pats.append(f"?{v} {ig.rel_str(preds[0])} ?{av}_n . ?{av}_n {ig.rel_str(preds[1])} ?{av} .")
        add_vars.append(av)
    if not add_vars:
        return None
    head, body = base_sparql.split("SELECT DISTINCT ", 1)
    proj, rest = body.split(" WHERE {", 1)
    rest = rest.rstrip().rstrip("}")
    return f"{head}SELECT DISTINCT {proj} {' '.join('?' + a for a in add_vars)} WHERE {{{rest}\n  " + "\n  ".join(add_pats) + "\n}"


def _run(store, sparql, ig, cap=2000):
    try:
        sols = store.query(sparql, use_default_graph_as_union=True)
        names = [v.value for v in sols.variables]
        rows = []
        for i, sol in enumerate(sols):
            if i >= cap:
                break
            rows.append([ig.pfx.c(kg.term_str(sol[n]))[:60] if sol[n] is not None else "" for n in names])
        return names, rows
    except Exception:  # noqa: BLE001
        return None, []


@torch.no_grad()
def gnn_candidates_v2(model, ig, question, store, n_cand=5, n_paths=12):
    q = embed([question]).to(ig.dev)
    out = bridge_scores(model, ig, q)
    nl = out["node_logit"][0]
    is_attr = ig.is_attr.to(nl.device)
    n_inst = ig.g["n_inst"]
    inst_mask = torch.zeros_like(nl, dtype=torch.bool)
    inst_mask[:n_inst] = True
    inst_mask &= ~is_attr
    seeds = [i for i in nl.masked_fill(~inst_mask, -1e9).topk(10).indices.tolist()]
    beams = beam_paths(model, ig, out, seeds)
    agg = {}
    for nodes, edges, sc in beams:
        k = lift_key(ig, nodes, edges)
        if any(k[i] < 0 for i in range(0, len(k), 2)):
            continue
        agg[k] = max(agg.get(k, -1e9), sc / max(1, len(edges)) ** 0.5)
    ranked = [k for k, _ in sorted(agg.items(), key=lambda kv: -kv[1])][:n_paths]
    ans_prob = torch.sigmoid(out["ans_logit"][0]).cpu()
    pt = ig.g["primary_type"]
    ent_ans = ans_prob.masked_fill(~inst_mask.cpu(), 0)
    ans_classes = {int(pt[i]) for i in ent_ans.topk(30).indices.tolist() if int(pt[i]) >= 0}
    attrs = attribute_columns(ig, ans_prob)
    cands, seen = [], set()
    for k in ranked:
        if tuple(reversed(k)) in seen:
            continue
        seen.add(k)
        pfx = {p: ns for p, ns in ig.pfx.items}
        base = compile_path_query(ig, k, ans_classes, pfx)
        aug = augment_query(ig, base, k, attrs)
        for sparql, kind in ((aug, "with GNN-selected attribute columns"), (base, "entities only")):
            if not sparql:
                continue
            names, rows = _run(store, sparql, ig)
            sig = (len(rows), frozenset(map(tuple, (sorted(r) for r in rows[:50]))))
            if rows and sig in seen:
                break  # same result as an earlier candidate (e.g. the inverse-direction path)
            if rows:
                seen.add(sig)
                cands.append({"sparql": sparql, "kind": kind, "n_rows": len(rows), "cols": names, "sample": rows[:3]})
                break
        if len(cands) >= n_cand:
            break
    return cands, attrs


def e8_block(model, ig, question, store, skeleton_block):
    cands, attrs = gnn_candidates_v2(model, ig, question, store)
    parts = ["## Candidate query skeletons proposed by the graph retriever (already executed on this graph)",
             "Ranked by the retriever's relevance to the question; attribute columns were chosen by the retriever's"
             " answer model. Use, combine or extend them; keep the columns the question asks for."]
    for i, c in enumerate(cands, 1):
        body = c["sparql"].split("SELECT", 1)[1]
        parts.append(f"### Candidate {i} ({c['kind']}): {c['n_rows']} rows; columns {', '.join('?' + x for x in c['cols'])}\n"
                     f"SELECT{body}\nsample: " + " | ".join(", ".join(r) for r in c["sample"]))
    block = skeleton_block + "\n\n" + "\n".join(parts)
    return block, {"tokens": n_tokens(block), "n_candidates": len(cands),
                   "n_with_attrs": sum(c["kind"].startswith("with") for c in cands),
                   "candidates": [c["sparql"] for c in cands]}
