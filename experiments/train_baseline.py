"""Train an encode-then-match GNN baseline with BRIDGE's data, losses, schedule and selection rule.

Mirrors bridge.train.main (same batches of 16 same-building questions, AdamW, grad clip 1.0,
early stopping on val path Hit@10 + 1e-3 * node R@50, patience 3); only the model differs.

python -m experiments.train_baseline --name compgcn --encoder compgcn
python -m experiments.train_baseline --name gcn_norel --encoder gcn
python -m experiments.train_baseline --name mlp_text --encoder mlp
Output: runs/ckpt_exp/{name}.pt, runs/ckpt_exp/{name}.jsonl
"""
import argparse
import json
import random
import time

import torch

from bridge import kg
from bridge.train import PREP, batch_loss, evaluate, load_split, seed_all, to_dev
from experiments import encoders

CKPT = kg.ROOT / "runs" / "ckpt_exp"


def to_dev_x(g, dev):
    t = to_dev(g, dev)
    t["is_class"] = g["is_class"].to(dev)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", default="compgcn", choices=["compgcn", "gcn", "mlp"])
    ap.add_argument("--comp", default="mult", choices=["mult", "sub"])
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    # --- identical defaults to bridge.train
    ap.add_argument("--lambda-path", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-siblings", type=int, default=16)
    ap.add_argument("--max-neg-edges", type=int, default=2048)
    ap.add_argument("--buildings-limit", type=int)
    ap.add_argument("--val-max-q", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    seed_all(args.seed)
    dev = torch.device(args.device)
    sp = kg.split()
    train = load_split(sp["train"], args.buildings_limit)
    val = load_split(sp["val"], args.buildings_limit)
    qemb = torch.load(PREP / "qemb.pt", weights_only=False)["emb"]
    cfg = vars(args) | {"n_train_bldg": len(train), "n_val_bldg": len(val)}
    model = encoders.build(cfg).to(dev)
    cfg["params"] = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {args.encoder} params={cfg['params']:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    CKPT.mkdir(parents=True, exist_ok=True)
    logf = open(CKPT / f"{args.name}.jsonl", "w")
    logf.write(json.dumps({"config": cfg}) + "\n"); logf.flush()

    # bridge.evaluate moves graphs with bridge.train.to_dev (no is_class); patch it in for our model
    import bridge.train as T
    T.to_dev = to_dev_x

    batches = [(gi, i) for gi, g in enumerate(train) for i in range(0, len(g["samples"]), args.batch)]
    best, bad, t0 = -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        rng = random.Random(args.seed + ep)
        for g in train:
            rng.shuffle(g["samples"])
        rng.shuffle(batches)
        agg, n, cache = {"node": 0, "edge": 0, "path": 0, "ans": 0, "total": 0}, 0, {}
        model.train()
        for step, (gi, i) in enumerate(batches):
            if gi not in cache:
                cache = {gi: to_dev_x(train[gi], dev)}
            batch = train[gi]["samples"][i:i + args.batch]
            if not batch:
                continue
            loss, parts, _ = batch_loss(model, cache[gi], batch, qemb, args, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1
            agg["total"] += float(loss.detach())
            for k in parts:
                agg[k] += parts[k]
            if step % 200 == 0:
                print(f"ep{ep} step{step}/{len(batches)} loss={float(loss):.3f} {time.time()-t0:.0f}s", flush=True)
        ev = evaluate(model, val, qemb, args, dev, args.val_max_q)
        rec = {"epoch": ep, "train": {k: v / max(n, 1) for k, v in agg.items()}, "val": ev,
               "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        score = ev["path_hit@10"] + 1e-3 * ev["node_recall@50"]
        if score > best:
            best, bad = score, 0
            torch.save({"state": model.state_dict(), "config": cfg, "epoch": ep, "val": ev}, CKPT / f"{args.name}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    logf.write(json.dumps({"done": True, "best_score": best, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    logf.close()


if __name__ == "__main__":
    main()
