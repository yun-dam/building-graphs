"""Build E8 context blocks (whole-building skeleton from contexts_E5 + v2 GNN query candidates).

python -m bridge.v2.contexts_e8 --dataset dev223 [--ckpt runs/v2/ckpt_vm/v2main.pt]
"""
import argparse
import collections
import json

import numpy as np
import pyoxigraph as ox
import torch

from bridge import kg, qsets
from bridge.retrieve import load_model
from bridge.v2.retrieve_v2 import e8_block, v2_bq_graph, v2_dev_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", default=str(kg.ROOT / "runs" / "v2" / "ckpt_vm" / "v2main.pt"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(a.ckpt, dev)
    cdir = qsets.contexts_dir(a.dataset)
    e5 = json.loads((cdir / "contexts_E5.json").read_text(encoding="utf-8"))
    by_b = collections.defaultdict(list)
    for q in qsets.questions(a.dataset):
        by_b[q["building"]].append(q)
    out = {}
    for b, items in by_b.items():
        ig = v2_bq_graph(b, dev) if a.dataset == "buildingqa" else v2_dev_graph(b, dev)
        store = kg.make_store(items[0]["ttl"], items[0]["tbox"], abox_format=ox.RdfFormat.TURTLE)
        for q in items:
            blk, info = e8_block(model, ig, q["text"], store, e5[q["qkey"]]["block"])
            out[q["qkey"]] = {"block": blk, **info}
        print(b, len(items), "E8 tokens", np.mean([out[q["qkey"]]["tokens"] for q in items]).round(0),
              "cands w/ attrs", sum(out[q["qkey"]]["n_with_attrs"] for q in items), flush=True)
    (cdir / "contexts_E8.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (cdir / "contexts_E8_meta.json").write_text(json.dumps({"ckpt": a.ckpt}))


if __name__ == "__main__":
    main()
