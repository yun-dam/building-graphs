"""BRIDGE-QBE on BuildingQA (no LLM). Config fixed on Build2SPARQL val buildings (tau=0.3, rho=0.6,
20 anchors, shortest-path patterns). BuildingQA graphs are used at inference only; gold rows only for scoring.

python -m bridge.qbe.eval_buildingqa_qbe
"""
import collections
import json
import time

import numpy as np
import torch

from bridge import kg, qsets
from bridge.qbe.eval_offline import score
from bridge.qbe.qbe import QBEGraph, qbe
from bridge.retrieve import load_model
from bridge.v2.retrieve_v2 import v2_bq_graph

ROOT = kg.ROOT
CFG = dict(tau=0.3, rho=0.6, n_anchor=20, shortest=True, max_hops=4, n_allowed=3000)


def main(ckpt=ROOT / "runs" / "v2" / "ckpt_vm" / "v2main.pt"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ckpt, dev)
    gold = qsets.gold("buildingqa")
    qs = qsets.questions("buildingqa")
    cache, per, t0 = {}, [], time.time()
    for q in qs:
        b = q["building"]
        if b not in cache:
            ig = v2_bq_graph(b, dev)
            tbox = kg.TBOX_223P if b == "b59" else kg.TBOX_BRICK
            cache = {b: (QBEGraph(ig), kg.make_store(kg.NTDIR / f"bq_{b}.nt", tbox))}
        qg, store = cache[b]
        t = time.time()
        r = qbe(model, qg, q["text"], store, **CFG)
        rows = (r or {}).get("rows") or []
        s = score(gold[q["qid"]], rows)
        s.update({"qkey": q["qkey"], "building": b, "source": q["source"], "latency_s": round(time.time() - t, 2),
                  "n_cols_pred": len((r or {}).get("columns", [])), "n_cols_gold": len(gold[q["qid"]][0]),
                  "n_rows_pred": len(rows), "sparql": (r or {}).get("sparql")})
        per.append(s)
    blds = ["TUC_building", "dflexlibs_multizone", "bldg11", "b59"]
    summ = {"config": CFG, "ckpt": str(ckpt)}
    for m in ("row_f1", "entity_f1", "exact_f1", "arity_f1"):
        summ[f"qw_{m}"] = float(np.mean([s[m] for s in per]))
        summ[f"macro_{m}"] = float(np.mean([np.mean([s[m] for s in per if s["building"] == b]) for b in blds]))
        for b in blds:
            summ[f"{b}_{m}"] = float(np.mean([s[m] for s in per if s["building"] == b]))
    n = {b: sum(1 for s in per if s["building"] == b) for b in blds}
    summ["brick3_qw_row_f1"] = sum(summ[f"{b}_row_f1"] * n[b] for b in blds[:3]) / sum(n[b] for b in blds[:3])
    by_src = collections.defaultdict(list)
    for s in per:
        by_src[s["source"]].append(s["row_f1"])
    summ["by_source_row_f1"] = {k: float(np.mean(v)) for k, v in by_src.items()}
    summ["mean_latency_s"] = float(np.mean([s["latency_s"] for s in per]))
    out = ROOT / "runs" / "qbe" / "buildingqa_qbe.json"
    out.write_text(json.dumps({"summary": summ, "per_question": per}, indent=1))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "config"}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
