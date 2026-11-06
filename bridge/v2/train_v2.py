"""Train BRIDGE v2 (same model / losses as v1: bridge.model.BRIDGE, bridge.train.batch_loss) on
runs/v2/prep (v2 graphs with attribute nodes; Build2SPARQL + synthetic multi-column samples).

Validation (val buildings):
  * path_hit@{1,5,10}, node_recall@{10,50}  on Build2SPARQL val samples (bridge.train.evaluate)
  * synthetic multi-column answer metrics on synthetic val samples:
      synth_ans_f1      = F1 of the top-|Y_ans| answer-head nodes (= R-precision; calibration-free)
      synth_ans_f1@0.5  = F1 of nodes with sigmoid(ans_logit) > 0.5
      synth_col_hit@20  = share of projected columns with >= 1 gold node among the top-20 answer nodes
      synth_litcol_hit@20 = same, literal-valued columns only
Early stopping / checkpoint selection on  path_hit@10 + es_ans_weight * synth_ans_f1.

Run from the repo root (local GPU or Linux VM):
  python -m bridge.v2.train_v2 --name v2main [--epochs 20 --d 64 --lambda-path 0.5 --synth-weight 1.0]
Set BRIDGE_BUCKET=gs://... to mirror checkpoints/logs to {bucket}/v2/ckpt/.
"""
import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch

from bridge import kg
from bridge.model import BRIDGE
from bridge.train import batch_loss, evaluate, seed_all, to_dev
from bridge.v2 import common as C


def load_split(names, limit=None, synth_weight=1.0, use_synth=True):
    out = []
    for b in (names[:limit] if limit else names):
        f = C.PREP / f"{b}.pt"
        if not f.exists():
            continue
        g = torch.load(f, weights_only=False)
        ss = []
        for s in g["samples"]:
            if s["source"] == "synth":
                if not use_synth:
                    continue
                s = dict(s, weight=s["weight"] * synth_weight)
            ss.append(s)
        g["samples"] = ss
        out.append(g)
    return out


def subset(graphs, source):
    return [dict(g, samples=[s for s in g["samples"] if s["source"] == source]) for g in graphs]


@torch.no_grad()
def eval_answers(model, graphs, qemb, args, dev, max_q=None):
    model.eval()
    m = {"synth_ans_f1": [], "synth_ans_f1@0.5": [], "synth_col_hit@20": [], "synth_litcol_hit@20": []}
    for g0 in graphs:
        ss = [s for s in g0["samples"] if s["source"] == "synth" and s["ans"]]
        if max_q:
            ss = ss[:max_q]
        if not ss:
            continue
        g = to_dev(g0, dev)
        N = g["x"].size(0)
        for i in range(0, len(ss), args.batch):
            batch = ss[i:i + args.batch]
            q = qemb[[s["q_idx"] for s in batch]].to(dev).float()
            out = model(g, q)
            for b, s in enumerate(batch):
                al = out["ans_logit"][b]
                Y = set(s["ans"])
                top = set(al.topk(min(len(Y), N)).indices.tolist())
                m["synth_ans_f1"].append(len(top & Y) / len(Y))
                pred = set((al > 0).nonzero().squeeze(-1).tolist())
                tp = len(pred & Y)
                m["synth_ans_f1@0.5"].append(2 * tp / (len(pred) + len(Y)) if (pred or Y) else 0.0)
                t20 = set(al.topk(min(20, N)).indices.tolist())
                for ci, col in enumerate(s["cols"]):
                    if not col:
                        continue
                    h = float(bool(t20 & set(col)))
                    m["synth_col_hit@20"].append(h)
                    if ci in s.get("lit_cols", []):
                        m["synth_litcol_hit@20"].append(h)
    model.train()
    r = {k: float(np.mean(v)) if v else 0.0 for k, v in m.items()}
    r["n_synth_q"] = len(m["synth_ans_f1"])
    return r


def gcs_mirror(path):
    bucket = os.environ.get("BRIDGE_BUCKET")
    if bucket:
        subprocess.run(["gsutil", "-q", "cp", str(path), f"{bucket.rstrip('/')}/v2/ckpt/"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2main")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ent-layers", type=int, default=6)
    ap.add_argument("--rel-layers", type=int, default=6)
    ap.add_argument("--lambda-path", type=float, default=0.5)
    ap.add_argument("--no-path-loss", action="store_true")
    ap.add_argument("--hard-seeding", action="store_true")
    ap.add_argument("--no-relation-graph", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-siblings", type=int, default=16)
    ap.add_argument("--max-neg-edges", type=int, default=2048)
    ap.add_argument("--synth-weight", type=float, default=1.0, help="loss weight multiplier for synthetic samples")
    ap.add_argument("--no-synth", action="store_true", help="train on Build2SPARQL samples only")
    ap.add_argument("--es-ans-weight", type=float, default=1.0, help="weight of synth_ans_f1 in the early-stopping score")
    ap.add_argument("--buildings", nargs="*", help="restrict to these buildings (smoke tests)")
    ap.add_argument("--buildings-limit", type=int)
    ap.add_argument("--val-max-q", type=int)
    ap.add_argument("--max-steps", type=int, help="stop each epoch after this many steps (smoke tests)")
    ap.add_argument("--init", help="optional checkpoint to initialise from (e.g. runs/ckpt/main.pt)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.no_path_loss:
        args.lambda_path = 0.0
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = kg.split()
    tr_names, va_names = sp["train"], sp["val"]
    if args.buildings:
        tr_names = [b for b in tr_names if b in args.buildings]
        va_names = [b for b in va_names if b in args.buildings]
    train = load_split(tr_names, args.buildings_limit, args.synth_weight, not args.no_synth)
    val = load_split(va_names, args.buildings_limit)
    val_b2s = subset(val, "b2s")
    qemb = torch.load(C.PREP / "qemb.pt", weights_only=False)["emb"]
    model = BRIDGE(d=args.d, rel_layers=args.rel_layers, ent_layers=args.ent_layers, dropout=args.dropout,
                   hard_seeding=args.hard_seeding, relation_graph=not args.no_relation_graph).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev, weights_only=False)["state"]
        print("init from", args.init, model.load_state_dict(sd, strict=False), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    C.CKPT.mkdir(parents=True, exist_ok=True)
    logf = open(C.CKPT / f"{args.name}.jsonl", "a")
    n_tr = {src: sum(1 for g in train for s in g["samples"] if s["source"] == src) for src in ("b2s", "synth")}
    cfg = vars(args) | {"version": 2, "n_train_bldg": len(train), "n_val_bldg": len(val), "n_train_samples": n_tr,
                        "init_note": args.init or "scratch"}
    print(json.dumps({"config": cfg}), flush=True)
    logf.write(json.dumps({"config": cfg}) + "\n"); logf.flush()
    batches = [(gi, i) for gi, g in enumerate(train) for i in range(0, len(g["samples"]), args.batch)]
    best, bad, t0 = -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        rng = random.Random(args.seed + ep)
        for g in train:
            rng.shuffle(g["samples"])
        rng.shuffle(batches)
        agg, n = {"node": 0, "edge": 0, "path": 0, "ans": 0, "total": 0}, 0
        cache = {}
        for step, (gi, i) in enumerate(batches):
            if args.max_steps and step >= args.max_steps:
                break
            if gi not in cache:
                cache = {gi: to_dev(train[gi], dev)}
            g = cache[gi]
            batch = train[gi]["samples"][i:i + args.batch]
            if not batch:
                continue
            loss, parts, _ = batch_loss(model, g, batch, qemb, args, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1
            agg["total"] += float(loss)
            for k in parts:
                agg[k] += parts[k]
            if step % 200 == 0:
                print(f"ep{ep} step{step}/{len(batches)} loss={float(loss):.3f} {parts} {time.time()-t0:.0f}s", flush=True)
        tr = {k: v / max(n, 1) for k, v in agg.items()}
        ev = evaluate(model, val_b2s, qemb, args, dev, args.val_max_q)
        ev.update(eval_answers(model, val, qemb, args, dev, args.val_max_q))
        rec = {"epoch": ep, "train": tr, "val": ev, "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        score = ev["path_hit@10"] + args.es_ans_weight * ev["synth_ans_f1"] + 1e-3 * ev["node_recall@50"]
        if score > best:
            best, bad = score, 0
            path = C.CKPT / f"{args.name}.pt"
            torch.save({"state": model.state_dict(), "config": cfg, "epoch": ep, "val": ev}, path)
            gcs_mirror(path)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    logf.write(json.dumps({"done": True, "best_score": best, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    logf.close()
    gcs_mirror(C.CKPT / f"{args.name}.jsonl")


if __name__ == "__main__":
    main()
