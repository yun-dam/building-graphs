"""Turn labels + graphs into training samples (one per question), with path candidates.

Cached to runs/prep/{building}.pt:
  graph tensors (+ rel_ids) and samples [{qid, reg, q_idx, nodes, edges, ans, weight, path}],
  question embeddings in runs/prep/qemb.pt
"""
import collections
import json
import random
import zlib
from pathlib import Path

import torch

from bridge import kg
from bridge.graph_build import embed

ROOT = kg.ROOT
PREP = ROOT / "runs" / "prep"
GDIR = ROOT / "runs" / "graphs"
LDIR = ROOT / "runs" / "labels"
MAX_PATH_LEN = 4
N_SIBLINGS = 16
N_EVAL_CANDS = 63
EDGES_PER_STEP = 32


def schema_graph(g):
    """Class-level graph over primary types of instance endpoints (ABox predicates only)."""
    pt = g["primary_type"]
    ei, et = g["edge_index"], g["edge_type"]
    names = g["rel_names"]
    skip = {i for i, n in enumerate(names) if n.lstrip("^") in (kg.RDF_TYPE, kg.SUBCLASS)}
    steps = collections.defaultdict(list)
    for e in range(ei.size(1)):
        r = int(et[e])
        if r in skip:
            continue
        a, b = int(pt[ei[0, e]]), int(pt[ei[1, e]])
        if a < 0 or b < 0:
            continue
        steps[(a, r, b)].append(e)
    adj = collections.defaultdict(set)
    for (a, r, b) in steps:
        adj[a].add((r, b))
    return steps, {k: sorted(v) for k, v in adj.items()}


def enum_paths(adj, start, max_len, cap=4000):
    """All class-level walks from start up to max_len (no immediate class repeats)."""
    out, stack = [], [((start,), ())]
    while stack and len(out) < cap:
        nodes, rels = stack.pop()
        if rels:
            out.append((nodes, rels))
        if len(rels) < max_len:
            for r, b in adj.get(nodes[-1], ()):
                if b in nodes:
                    continue
                stack.append((nodes + (b,), rels + (r,)))
    return out


def path_key(nodes, rels):
    return tuple(x for pair in zip(nodes, rels + (None,)) for x in pair)[:-1]


def gold_path(rec, g, cidx, pid):
    """P_inst = [C0, p1, d1, C1, ..., pk, dk, Ck] -> (class node ids, relation ids)."""
    p = rec.get("P_inst")
    if not p or (len(p) - 1) % 3:
        return None
    cls, preds, dirs = p[0::3], p[1::3], p[2::3]
    if any(c not in cidx for c in cls) or any(x not in pid for x in preds) or len(preds) > MAX_PATH_LEN:
        return None
    return tuple(cidx[c] for c in cls), tuple(2 * pid[x] + (0 if d == 1 else 1) for x, d in zip(preds, dirs))


def candidates(adj, steps, gold, rng, n):
    nodes, rels = gold
    L = len(rels)
    allp = enum_paths(adj, nodes[0], min(MAX_PATH_LEN, L + 1))
    gk = path_key(nodes, rels)
    same_end = [p for p in allp if p[0][-1] == nodes[-1] and path_key(*p) != gk]
    other = [p for p in allp if p[0][-1] != nodes[-1]]
    rng.shuffle(same_end)
    rng.shuffle(other)
    sib = (same_end + other)[:n]
    return [p for p in sib if all((p[0][i], p[1][i], p[0][i + 1]) in steps for i in range(len(p[1])))]


def path_edges(path, steps, rng):
    nodes, rels = path
    out = []
    for i, r in enumerate(rels):
        es = steps.get((nodes[i], r, nodes[i + 1]), [])
        if len(es) > EDGES_PER_STEP:
            es = rng.sample(es, EDGES_PER_STEP)
        out.append(es)
    return out


def prep_building(b, qtext_index):
    g = torch.load(GDIR / f"{b}.pt", weights_only=False)
    lab = json.loads((LDIR / f"{b}.json").read_text())
    idx = {n: i for i, n in enumerate(g["node_ids"])}
    names = g["rel_names"]
    preds = [n for n in names if not n.startswith("^")]
    pid = {p: i for i, p in enumerate(preds)}
    cidx = {n: idx[n] for n in g["node_ids"][g["n_inst"]:]}
    E = g["n_fwd_edges"]
    ei, et = g["edge_index"], g["edge_type"]
    emap = {(int(ei[0, e]), int(et[e]), int(ei[1, e])): e for e in range(E)}
    steps, adj = schema_graph(g)
    rng = random.Random(0)
    samples = []
    for rec in lab["queries"]:
        nodes = sorted({idx[n] for n in rec["Y_V"] if n in idx})
        edges = sorted({emap[k] for s, p, o in rec["Y_E"]
                        if s in idx and o in idx and p in pid
                        and (k := (idx[s], 2 * pid[p], idx[o])) in emap})
        ans = sorted({idx[n] for n in rec["Y_ans"] if n in idx})
        if not nodes:
            continue
        path = None
        gp = gold_path(rec, g, cidx, pid)
        if gp and all((gp[0][i], gp[1][i], gp[0][i + 1]) in steps for i in range(len(gp[1]))):
            cands = candidates(adj, steps, gp, rng, N_EVAL_CANDS)
            allp = [gp] + cands
            path = {"paths": [(list(p[0]), list(p[1])) for p in allp],
                    "edges": [path_edges(p, steps, rng) for p in allp]}
        for reg, qt in enumerate(rec["questions"]):
            samples.append({"qid": rec["query_id"], "reg": reg + 1, "q_idx": qtext_index[qt],
                            "nodes": nodes, "edges": edges, "ans": ans, "weight": rec["weight"],
                            "path": path, "pattern": rec["pattern_type"]})
    rel_ids = torch.tensor([zlib.crc32(n.encode()) % 4096 for n in names], dtype=torch.long)
    keep = ["x", "edge_index", "edge_type", "rel_x", "rel_graph", "rel_graph_type", "n_fwd_edges",
            "n_inst", "is_class", "primary_type", "stratum"]
    out = {k: g[k] for k in keep}
    out["rel_ids"] = rel_ids
    out["building"] = b
    out["samples"] = samples
    return out


def main():
    PREP.mkdir(parents=True, exist_ok=True)
    man = kg.manifest()
    texts = []
    for b in man:
        lab = json.loads((LDIR / f"{b}.json").read_text())
        for rec in lab["queries"]:
            texts += rec["questions"]
    texts = sorted(set(texts))
    qtext_index = {t: i for i, t in enumerate(texts)}
    torch.save({"texts": texts, "emb": embed(texts).half()}, PREP / "qemb.pt")
    stats = collections.Counter()
    for b in sorted(man):
        d = prep_building(b, qtext_index)
        torch.save(d, PREP / f"{b}.pt")
        stats["samples"] += len(d["samples"])
        stats["with_path"] += sum(1 for s in d["samples"] if s["path"])
        stats["no_edges"] += sum(1 for s in d["samples"] if not s["edges"])
    print(dict(stats))


if __name__ == "__main__":
    main()
