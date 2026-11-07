"""Mean ± std over seeds for experiments.eval_baseline outputs.

Model names in the per-seed files may carry a `_s{k}` suffix; it is stripped so the same model
is grouped across seeds (seed 0 files have no suffix).

python -m experiments.aggregate_seeds runs/exp/retriever_eval_baselines.json \
    runs/exp/retriever_eval_baselines_s1.json runs/exp/retriever_eval_baselines_s2.json
Output: runs/exp/retriever_eval_seeds.json and a markdown table on stdout.
"""
import argparse
import collections
import json
import re
import statistics

from experiments.eval_baseline import COLS, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--strata", nargs="*", default=["all", "s223"])
    ap.add_argument("--out", default=str(OUT / "retriever_eval_seeds.json"))
    a = ap.parse_args()
    vals = collections.defaultdict(lambda: collections.defaultdict(list))  # (model, stratum) -> col -> [v]
    params = {}
    for f in a.files:
        for name, v in json.load(open(f)).items():
            m = re.sub(r"_s\d+$", "", name)
            params[m] = v["params"]
            for st, r in v.items():
                if isinstance(r, dict):
                    for c, x in r.items():
                        if isinstance(x, (int, float)):
                            vals[(m, st)][c].append(float(x))
    summary = {}
    for (m, st), cols in vals.items():
        summary.setdefault(m, {"params": params[m]})[st] = {
            c: {"mean": statistics.mean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}
            for c, xs in cols.items()}
    json.dump(summary, open(a.out, "w"), indent=1)
    for st in a.strata:
        print(f"\n### {st}\n\n| model | params | seeds | " + " | ".join(COLS) + " |")
        print("|---|---:|---:|" + "---:|" * len(COLS))
        for m, v in summary.items():
            r = v.get(st)
            if not r:
                continue
            n = r[COLS[0]]["n"]
            cells = [f"{r[c]['mean']:.3f} ± {r[c]['std']:.3f}" if c in r else "–" for c in COLS]
            print(f"| {m} | {v['params']:,} | {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
