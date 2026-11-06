"""BRIDGE v2 dataset prep: like bridge.data but on v2 graphs, v2 labels (+ synth for train/val).

Each sample additionally has  source: b2s|synth  and  cols: [[node ids] per projected column]
(for multi-column answer metrics). Synthetic samples are only built for TRAIN and VAL buildings.

python -m bridge.v2.data_v2 [--buildings ...] [--split train val test]
Output: runs/v2/prep/{b}.pt, runs/v2/prep/qemb.pt
"""
import argparse
import collections
import json
import random
import time
import zlib

import torch

from bridge import kg
from bridge.data import N_EVAL_CANDS, candidates, gold_path, path_edges, schema_graph
from bridge.graph_build import embed
from bridge.v2 import common as C

KEEP = ["x", "edge_index", "edge_type", "rel_x", "rel_graph", "rel_graph_type", "n_fwd_edges",
        "n_inst", "n_attr", "is_class", "is_attr", "primary_type", "stratum"]


def records(b, split):
    out = []
    f = C.LDIR / f"{b}.json"
    if f.exists():
        out += [dict(r, source="b2s") for r in json.loads(f.read_text())["queries"]]
    if split in ("train", "val"):
        f = C.SDIR / f"{b}.json"
        if f.exists():
            out += [dict(r, source="synth") for r in json.loads(f.read_text())["queries"] if r.get("questions")]
    return out


def prep_building(b, recs, qtext_index):
    g = torch.load(C.GDIR / f"{b}.pt", weights_only=False)
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
    for rec in recs:
        nodes = sorted({idx[n] for n in rec["Y_V"] if n in idx})
        edges = sorted({emap[k] for s, p, o in rec["Y_E"]
                        if s in idx and o in idx and p in pid
                        and (k := (idx[s], 2 * pid[p], idx[o])) in emap})
        ans = sorted({idx[n] for n in rec["Y_ans"] if n in idx})
        if not nodes:
            continue
        cols = [sorted({idx[n] for n in rec.get("Y_ans_cols", {}).get(v, []) if n in idx}) for v in rec["proj"]]
        path = None
        gp = gold_path(rec, g, cidx, pid)
        if gp and all((gp[0][i], gp[1][i], gp[0][i + 1]) in steps for i in range(len(gp[1]))):
            cands = candidates(adj, steps, gp, rng, N_EVAL_CANDS)
            allp = [gp] + cands
            path = {"paths": [(list(p[0]), list(p[1])) for p in allp],
                    "edges": [path_edges(p, steps, rng) for p in allp]}
        for reg, qt in enumerate(rec["questions"]):
            samples.append({"qid": rec["query_id"], "reg": reg + 1, "q_idx": qtext_index[qt],
                            "nodes": nodes, "edges": edges, "ans": ans, "cols": cols,
                            "weight": rec.get("weight", 1.0), "path": path,
                            "pattern": rec["pattern_type"], "source": rec["source"],
                            "lit_cols": [i for i, v in enumerate(rec["proj"]) if v in rec.get("lit_vars", [])]})
    rel_ids = torch.tensor([zlib.crc32(n.encode()) % 4096 for n in names], dtype=torch.long)
    out = {k: g[k] for k in KEEP}
    out["rel_ids"] = rel_ids
    out["building"] = b
    out["samples"] = samples
    return out


def main(buildings=None, splits=("train", "val", "test")):
    C.PREP.mkdir(parents=True, exist_ok=True)
    sp = kg.split()
    todo = buildings or [b for k in splits for b in sp[k]]
    todo = [b for b in todo if (C.GDIR / f"{b}.pt").exists()]
    recs = {b: records(b, C.split_of(b)) for b in todo}
    # question embeddings: union with an existing qemb so partial re-preps keep indices valid
    qf = C.PREP / "qemb.pt"
    old = torch.load(qf, weights_only=False) if qf.exists() else {"texts": [], "emb": torch.zeros(0, 384).half()}
    have = set(old["texts"])
    new = sorted({q for rs in recs.values() for r in rs for q in r["questions"]} - have)
    texts = old["texts"] + new
    emb = torch.cat([old["emb"], embed(new).half()]) if new else old["emb"]
    torch.save({"texts": texts, "emb": emb}, qf)
    qtext_index = {t: i for i, t in enumerate(texts)}
    stats = collections.defaultdict(collections.Counter)
    t0 = time.time()
    for b in todo:
        d = prep_building(b, recs[b], qtext_index)
        torch.save(d, C.PREP / f"{b}.pt")
        st = stats[C.split_of(b)]
        for s in d["samples"]:
            st[f"{s['source']}_samples"] += 1
            st[f"{s['source']}_with_path"] += bool(s["path"])
            st[f"{s['source']}_lit_ans"] += bool(s["lit_cols"])
    print(f"prepped {len(todo)} buildings in {time.time()-t0:.0f}s; qemb {len(texts)} (+{len(new)})")
    for k, v in stats.items():
        print(k, dict(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val", "test"])
    a = ap.parse_args()
    main(a.buildings, tuple(a.split))
