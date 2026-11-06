"""Train BRIDGE v3 on runs/hybrid/prep (ontology-filtered graphs; Build2SPARQL + v2 synth + long synth).

Same model / losses / optimisation as bridge.v2.train_v2 (bridge.model.BRIDGE, bridge.train.batch_loss);
defaults = v2main except  --ent-layers 8  (deeper entity NBFNet for 6-7 hop BuildingQA-style queries).

Validation (val buildings):
  path_hit@{1,5,10}, node_recall@{10,50}     Build2SPARQL val samples (bridge.train.evaluate)
  synth_ans_f1, synth_col_hit@20, ...          v2 synthetic val samples (bridge.v2.train_v2.eval_answers)
  long_ans_f1, long_litcol_hit@20, ...         long synthetic val samples (same metrics)
  long_path_hit@{1,5,10}                       gold-path Hit over ALL val samples with >= 4 hops
                                               (non-type triple patterns, literal tails included)
Early stopping score = path_hit@10 + es_ans_weight * synth_ans_f1 + es_long_weight * long_path_hit@10
                       (+ 1e-3 node_recall@50).

python -m bridge.hybrid.train_v3 --name v3main                      # 8 entity layers
python -m bridge.hybrid.train_v3 --name v3l6 --ent-layers 6         # v2 depth, for comparison
Smoke:  python -m bridge.hybrid.train_v3 --name smoke --buildings A B C VAL --epochs 1 --max-steps 30 --val-max-q 40
BRIDGE_BUCKET=gs://... mirrors checkpoints/logs to {bucket}/hybrid/ckpt/.
"""
import argparse
import json
import os
import random
import subprocess
import time

import torch

from bridge import kg
from bridge.hybrid import common as H
from bridge.model import BRIDGE
from bridge.train import batch_loss, evaluate, seed_all, to_dev
from bridge.v2.train_v2 import eval_answers


def load_split(names, limit=None, synth_weight=1.0, long_weight=1.0, use_synth=True):
    out = []
    for b in (names[:limit] if limit else names):
        f = H.PREP / f"{b}.pt"
        if not f.exists():
            continue
        g = torch.load(f, weights_only=False)
        ss = []
        for s in g["samples"]:
            if s["source"] in ("synth", "synth_long"):
                if not use_synth:
                    continue
                s = dict(s, weight=s["weight"] * (synth_weight if s["source"] == "synth" else long_weight))
            ss.append(s)
        g["samples"] = ss
        out.append(g)
    return out


def subset(graphs, pred):
    return [dict(g, samples=[s for s in g["samples"] if pred(s)]) for g in graphs]


def gcs_mirror(path):
    bucket = os.environ.get("BRIDGE_BUCKET")
    if bucket:
        subprocess.run(["gsutil", "-q", "cp", str(path), f"{bucket.rstrip('/')}/hybrid/ckpt/"], check=False)


def validate(model, val, qemb, args, dev):
    ev = evaluate(model, subset(val, lambda s: s["source"] == "b2s"), qemb, args, dev, args.val_max_q)
    ev.update(eval_answers(model, val, qemb, args, dev, args.val_max_q))
    # long synthetic answers: reuse eval_answers by presenting long samples as "synth"
    long = subset(val, lambda s: s["source"] == "synth_long")
    long = [dict(g, samples=[dict(s, source="synth") for s in g["samples"]]) for g in long]
    ev.update({k.replace("synth_", "long_"): v for k, v in eval_answers(model, long, qemb, args, dev, args.val_max_q).items()})
    lp = evaluate(model, subset(val, lambda s: s.get("n_hops", 0) >= 4 and s["path"] is not None), qemb, args, dev,
                  args.val_max_q)
    ev.update({"long_" + k: v for k, v in lp.items() if k.startswith("path_hit") or k == "n_path_q"})
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v3main")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ent-layers", type=int, default=8)
    ap.add_argument("--rel-layers", type=int, default=6)
    ap.add_argument("--lambda-path", type=float, default=0.5)
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
    ap.add_argument("--synth-weight", type=float, default=1.0)
    ap.add_argument("--long-weight", type=float, default=1.0, help="loss weight multiplier for long synthetic samples")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--es-ans-weight", type=float, default=1.0)
    ap.add_argument("--es-long-weight", type=float, default=1.0)
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--buildings-limit", type=int)
    ap.add_argument("--val-max-q", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--init")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.no_path_loss = args.lambda_path == 0
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = kg.split()
    tr_names, va_names = sp["train"], sp["val"]
    if args.buildings:
        tr_names = [b for b in tr_names if b in args.buildings]
        va_names = [b for b in va_names if b in args.buildings]
    train = load_split(tr_names, args.buildings_limit, args.synth_weight, args.long_weight, not args.no_synth)
    val = load_split(va_names, args.buildings_limit)
    qemb = torch.load(H.PREP / "qemb.pt", weights_only=False)["emb"]
    model = BRIDGE(d=args.d, rel_layers=args.rel_layers, ent_layers=args.ent_layers, dropout=args.dropout,
                   hard_seeding=args.hard_seeding, relation_graph=not args.no_relation_graph).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev, weights_only=False)["state"]
        print("init from", args.init, model.load_state_dict(sd, strict=False), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    H.CKPT.mkdir(parents=True, exist_ok=True)
    logf = open(H.CKPT / f"{args.name}.jsonl", "a")
    n_tr = {src: sum(1 for g in train for s in g["samples"] if s["source"] == src) for src in ("b2s", "synth", "synth_long")}
    cfg = vars(args) | {"version": 3, "n_train_bldg": len(train), "n_val_bldg": len(val), "n_train_samples": n_tr,
                        "init_note": args.init or "scratch", "graphs": "hybrid (ontology-filtered)"}
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
                print(f"ep{ep} step{step}/{len(batches)} loss={float(loss):.3f} {parts} {time.time() - t0:.0f}s", flush=True)
        tr = {k: v / max(n, 1) for k, v in agg.items()}
        ev = validate(model, val, qemb, args, dev)
        rec = {"epoch": ep, "train": tr, "val": ev, "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        score = (ev["path_hit@10"] + args.es_ans_weight * ev["synth_ans_f1"]
                 + args.es_long_weight * ev["long_path_hit@10"] + 1e-3 * ev["node_recall@50"])
        if score > best:
            best, bad = score, 0
            path = H.CKPT / f"{args.name}.pt"
            torch.save({"state": model.state_dict(), "config": cfg, "epoch": ep, "val": ev}, path)
            gcs_mirror(path)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    logf.write(json.dumps({"done": True, "best_score": best, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    logf.close()
    gcs_mirror(H.CKPT / f"{args.name}.jsonl")


if __name__ == "__main__":
    main()
