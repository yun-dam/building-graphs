"""Offline (no LLM) evaluation of BRIDGE-QBE on the Build2SPARQL-derived question sets.

python -m bridge.qbe.eval_offline --dataset devval --taus 0.3 0.4 0.5 0.6 0.7     # tune on val buildings
python -m bridge.qbe.eval_offline --dataset dev223 --taus 0.5                       # held-out dev
"""
import argparse
import collections
import contextlib
import io
import json
import sys
import time

import numpy as np
import torch

from bridge import kg
from bridge.qbe.qbe import QBEGraph, qbe
from bridge.retrieve import load_model
from bridge.v2.retrieve_v2 import v2_dev_graph

ROOT = kg.ROOT
sys.path.insert(0, str(ROOT / "data" / "BuildingQA"))
import metrics as M  # noqa: E402

SETS = {"devval": "dev_setval", "dev": "dev_set", "dev223": "dev_set223"}


def score(gold, pred):
    if not pred:
        return {"arity_f1": 0.0, "entity_f1": 0.0, "row_f1": 0.0, "exact_f1": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        em = M.get_entity_and_row_matching_f1(gold, pred)
        return {"arity_f1": float(M.get_arity_matching_f1(pred, gold)), "entity_f1": float(em["entity_set_f1"]),
                "row_f1": float(em["row_matching_f1"]), "exact_f1": float(M.get_exact_match_f1(gold, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SETS))
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5])
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--n-allowed", type=int, default=3000)
    ap.add_argument("--rhos", nargs="+", type=float, default=[None])
    ap.add_argument("--n-anchor", type=int, default=3)
    ap.add_argument("--shortest", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "v2" / "ckpt_vm" / "v2main.pt"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(a.ckpt, dev)
    items = json.loads((ROOT / "runs" / "dev" / f"{SETS[a.dataset]}.json").read_text(encoding="utf-8"))
    cache = {}
    combos = [(t, r) for t in a.taus for r in a.rhos]
    res = {c: [] for c in combos}
    t0 = time.time()
    for it in items:
        b = it["building"]
        if b not in cache:
            ig = v2_dev_graph(b, dev)
            g = ig.g
            nt = kg.NTDIR / f"{g['building']}{'@compiled' if g.get('graph_variant') == 'compiled' else ''}.nt"
            tbox = kg.TBOX_223P if it["stratum"] == "s223" else kg.TBOX_BRICK
            cache = {b: (QBEGraph(ig), kg.make_store(nt, tbox))}
        qg, store = cache[b]
        for (t, rho) in combos:
            r = qbe(model, qg, it["text"], store, tau=t, max_hops=a.max_hops, n_allowed=a.n_allowed,
                    rho=rho, n_anchor=a.n_anchor, shortest=a.shortest)
            rows = (r or {}).get("rows") or []
            s = score(it["gold_rows"], rows)
            s.update({"qkey": it["qkey"], "stratum": it["stratum"], "n_cols_pred": len((r or {}).get("columns", [])),
                      "n_cols_gold": len(it["gold_rows"][0]), "sparql": (r or {}).get("sparql")})
            res[(t, rho)].append(s)
    summary = {}
    for t, rs in res.items():
        by = collections.defaultdict(list)
        for s in rs:
            by[s["stratum"]].append(s)
        summary[f"tau={t[0]},rho={t[1]}"] = {"all": {m: float(np.mean([s[m] for s in rs])) for m in ("row_f1", "entity_f1", "exact_f1", "arity_f1")},
                           "by_stratum": {k: float(np.mean([s["row_f1"] for s in v])) for k, v in by.items()},
                           "arity_exact_rate": float(np.mean([s["n_cols_pred"] == s["n_cols_gold"] for s in rs]))}
        key = f"tau={t[0]},rho={t[1]}"
        print(key, {k: round(v, 3) for k, v in summary[key]["all"].items()},
              {k: round(v, 3) for k, v in summary[key]["by_stratum"].items()},
              "arity==gold %.2f" % summary[key]["arity_exact_rate"], flush=True)
    out = ROOT / "runs" / "qbe" / f"offline_{a.dataset}{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "per_question": {f"tau={t[0]},rho={t[1]}": rs for t, rs in res.items()}}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
