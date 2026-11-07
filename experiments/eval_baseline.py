"""Test-split retriever evaluation for experiments/ baselines, with bridge.eval_retriever's metrics.

Uses bridge.eval_retriever.eval_bridge + summarize unchanged (node R@k, edge R@k inside the
3-hop region of the top-64 nodes, path Hit@k, answer F1 at sigmoid > 0.5), per stratum.
Context-token metrics are not computed (they need BRIDGE's verbalizer).

python -m experiments.eval_baseline --models compgcn gcn_norel mlp_text [--bridge v1_repro]
Output: runs/exp/retriever_eval_baselines.json and a markdown table on stdout.
"""
import argparse
import collections
import json
import time

import torch

from bridge import kg
from bridge.eval_retriever import PREP, eval_bridge, summarize
from bridge.retrieve import InfGraph, Prefixes, load_model
from experiments import encoders

ROOT = kg.ROOT
OUT = ROOT / "runs" / "exp"
COLS = ["node_recall@10", "node_recall@50", "edge_recall@50", "edge_recall@200",
        "path_hit@1", "path_hit@10", "answer_f1"]


def table(summary, strata=("all", "s223")):
    lines = []
    for st in strata:
        lines += [f"\n### {st}\n", "| model | params | " + " | ".join(COLS) + " |",
                  "|---|---:|" + "---:|" * len(COLS)]
        for m, v in summary.items():
            r = v.get(st)
            if not r:
                continue
            cells = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else "–" for c in COLS]
            lines.append(f"| {m} | {v['params']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["compgcn", "gcn_norel", "mlp_text"])
    ap.add_argument("--bridge", nargs="*", default=[], help="BRIDGE checkpoints in runs/ckpt to include")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT / "retriever_eval_baselines.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)
    man = kg.manifest()
    qemb = torch.load(PREP / "qemb.pt", weights_only=False)["emb"]
    models, params = {}, {}
    for name in a.bridge:
        models[name] = load_model(ROOT / "runs" / "ckpt" / f"{name}.pt", dev)[0]
    for name in a.models:
        models[name] = encoders.load(ROOT / "runs" / "ckpt_exp" / f"{name}.pt", dev)[0]
    for name, m in models.items():
        params[name] = sum(p.numel() for p in m.parameters())
    rows = {m: collections.defaultdict(list) for m in models}
    t0 = time.time()
    for b in kg.split()[a.split]:
        g = torch.load(ROOT / "runs" / "graphs" / f"{b}.pt", weights_only=False)
        prep = torch.load(PREP / f"{b}.pt", weights_only=False)
        g["rel_ids"] = prep["rel_ids"]
        ig = InfGraph(g, Prefixes(ROOT / man[b]["path"]), dev)
        ig.t["is_class"] = prep["is_class"].to(dev)
        strat = man[b]["stratum"]
        for name, model in models.items():
            for r in eval_bridge(model, ig, prep["samples"], qemb, dev, False):
                rows[name][strat].append(r)
        print(f"{b} ({strat}) done {time.time()-t0:.0f}s", flush=True)
    summary = {}
    for m in models:
        allr = [r for v in rows[m].values() for r in v]
        summary[m] = {"params": params[m], "all": summarize(allr), **{k: summarize(v) for k, v in rows[m].items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(table(summary))


if __name__ == "__main__":
    main()
