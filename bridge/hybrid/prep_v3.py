"""BRIDGE v3 dataset prep on the hybrid (ontology-filtered) graphs.

Sources per building:
  b2s         runs/hybrid/labels/{b}.json         (Build2SPARQL queries, v2 labels on hybrid graphs)
  synth       runs/hybrid/synth_v2/{b}.json       (v2 short multi-column synth, relabelled; train/val only)
  synth_long  runs/hybrid/synth_long/{b}.json     (3-7 hop synth with paraphrases; train/val only)
              falls back to runs/hybrid/synth_long_raw/{b}.json (template question only) when the
              paraphrase step has not been run, so training can start before paraphrasing.
Same sample format as bridge.v2.data_v2 plus  n_hops  (non-type triple patterns of the query) and
path_len; gold-path labels / candidates allow paths up to MAX_PATH_LEN = 7 class hops (v1/v2: 4).

python -m bridge.hybrid.prep_v3 [--buildings ...] [--split train val test] [--template-only]
Output: runs/hybrid/prep/{b}.pt, runs/hybrid/prep/qemb.pt
"""
import argparse
import collections
import json
import random
import time
import zlib

import torch

from bridge import kg
from bridge.data import N_EVAL_CANDS, path_edges, path_key, schema_graph
from bridge.graph_build import embed
from bridge.hybrid import common as H
from bridge.v2 import common as C2
from bridge.v2.data_v2 import KEEP

MAX_PATH_LEN = 7


def enum_paths(adj, start, max_len, cap=4000):
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


def gold_path(rec, cidx, pid):
    p = rec.get("P_inst")
    if not p or (len(p) - 1) % 3:
        return None
    cls, preds, dirs = p[0::3], p[1::3], p[2::3]
    if any(c not in cidx for c in cls) or any(x not in pid for x in preds) or len(preds) > MAX_PATH_LEN:
        return None
    return tuple(cidx[c] for c in cls), tuple(2 * pid[x] + (0 if d == 1 else 1) for x, d in zip(preds, dirs))


def candidates(adj, steps, gold, rng, n):
    """Same as bridge.data.candidates but up to MAX_PATH_LEN; siblings of the same length are mixed in
    so long gold paths compete with long negatives."""
    nodes, rels = gold
    L = len(rels)
    allp = enum_paths(adj, nodes[0], min(MAX_PATH_LEN, L + 1))
    gk = path_key(nodes, rels)
    same_end = [p for p in allp if p[0][-1] == nodes[-1] and path_key(*p) != gk]
    same_len = [p for p in allp if p[0][-1] != nodes[-1] and len(p[1]) == L]
    other = [p for p in allp if p[0][-1] != nodes[-1] and len(p[1]) != L]
    for x in (same_end, same_len, other):
        rng.shuffle(x)
    sib = (same_end + same_len[: n // 3] + other + same_len[n // 3:])[:n]
    return [p for p in sib if all((p[0][i], p[1][i], p[0][i + 1]) in steps for i in range(len(p[1])))]


def _load(path, source, template_only=False):
    if not path.exists():
        return []
    out = []
    for r in json.loads(path.read_text())["queries"]:
        qs = r.get("questions") or [r["template"]]
        if template_only and r.get("template"):
            qs = [r["template"]]
        n_hops = len(r.get("P_ast") or [])     # non-type triple patterns incl. literal tails
        out.append(dict(r, source=source, questions=qs, n_hops=n_hops))
    return out


def records(b, split, template_only=False):
    out = _load(H.LDIR / f"{b}.json", "b2s")
    if split in ("train", "val"):
        out += _load(H.H / "synth_v2" / f"{b}.json", "synth", template_only)
        f = H.SDIR / f"{b}.json"
        out += _load(f if f.exists() and not template_only else H.SRAW / f"{b}.json", "synth_long", template_only)
    return out


def prep_building(b, recs, qtext_index):
    g = torch.load(H.GDIR / f"{b}.pt", weights_only=False)
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
                        if s in idx and o in idx and p in pid and (k := (idx[s], 2 * pid[p], idx[o])) in emap})
        ans = sorted({idx[n] for n in rec["Y_ans"] if n in idx})
        if not nodes:
            continue
        cols = [sorted({idx[n] for n in rec.get("Y_ans_cols", {}).get(v, []) if n in idx}) for v in rec["proj"]]
        path, plen = None, 0
        gp = gold_path(rec, cidx, pid)
        if gp and all((gp[0][i], gp[1][i], gp[0][i + 1]) in steps for i in range(len(gp[1]))):
            allp = [gp] + candidates(adj, steps, gp, rng, N_EVAL_CANDS)
            path = {"paths": [(list(p[0]), list(p[1])) for p in allp],
                    "edges": [path_edges(p, steps, rng) for p in allp]}
            plen = len(gp[1])
        for reg, qt in enumerate(rec["questions"]):
            samples.append({"qid": rec["query_id"], "reg": reg + 1, "q_idx": qtext_index[qt],
                            "nodes": nodes, "edges": edges, "ans": ans, "cols": cols,
                            "weight": rec.get("weight", 1.0), "path": path, "path_len": plen,
                            "n_hops": rec["n_hops"], "pattern": rec["pattern_type"], "source": rec["source"],
                            "lit_cols": [i for i, v in enumerate(rec["proj"]) if v in rec.get("lit_vars", [])]})
    out = {k: g[k] for k in KEEP}
    out["rel_ids"] = torch.tensor([zlib.crc32(n.encode()) % 4096 for n in names], dtype=torch.long)
    out["building"] = b
    out["samples"] = samples
    return out


def main(buildings=None, splits=("train", "val", "test"), template_only=False):
    H.PREP.mkdir(parents=True, exist_ok=True)
    sp = kg.split()
    todo = buildings or [b for k in splits for b in sp[k]]
    todo = [b for b in todo if (H.GDIR / f"{b}.pt").exists()]
    recs = {b: records(b, C2.split_of(b), template_only) for b in todo}
    qf = H.PREP / "qemb.pt"
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
        torch.save(d, H.PREP / f"{b}.pt")
        st = stats[C2.split_of(b)]
        for s in d["samples"]:
            st[f"{s['source']}_samples"] += 1
            st[f"{s['source']}_with_path"] += bool(s["path"])
            st[f"{s['source']}_hops>=4"] += s["n_hops"] >= 4
            st[f"{s['source']}_hops>=4_with_path"] += s["n_hops"] >= 4 and bool(s["path"])
            st[f"{s['source']}_lit_ans"] += bool(s["lit_cols"])
    print(f"prepped {len(todo)} buildings in {time.time() - t0:.0f}s; qemb {len(texts)} (+{len(new)})")
    for k, v in stats.items():
        print(k, dict(sorted(v.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--template-only", action="store_true",
                    help="use only template questions for synthetic data (ignore paraphrases)")
    a = ap.parse_args()
    main(a.buildings, tuple(a.split), a.template_only)
