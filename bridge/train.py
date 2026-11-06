"""Step 5: train BRIDGE. Early stopping on val gold-path Hit@10.

python -m bridge.train --name main [--d 64 --ent-layers 6 --rel-layers 6 --lambda-path 0.5]
       [--hard-seeding] [--no-relation-graph] [--no-path-loss] [--epochs 20] [--buildings-limit N]
"""
import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from bridge import kg
from bridge.model import BRIDGE

ROOT = kg.ROOT
PREP = ROOT / "runs" / "prep"
CKPT = ROOT / "runs" / "ckpt"
GT_KEYS = ["x", "edge_index", "edge_type", "rel_x", "rel_graph", "rel_graph_type", "rel_ids"]


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(names, limit=None):
    out = []
    for b in names[:limit] if limit else names:
        f = PREP / f"{b}.pt"
        if f.exists():
            out.append(torch.load(f, weights_only=False))
    return out


def to_dev(g, dev):
    return {k: (g[k].to(dev) if k in GT_KEYS else g[k]) for k in g}


def bce_pw(logit, target, mask=None, max_pw=50.0):
    if mask is not None:
        logit, target = logit[mask], target[mask]
    pos = target.sum()
    if pos == 0:
        return logit.sum() * 0
    pw = ((target.numel() - pos) / pos).clamp(1.0, max_pw)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)


def edge_region(out_b, g, n_top=64, hops=3):
    top = out_b.topk(min(n_top, out_b.numel())).indices
    _, _, _, emask = k_hop_subgraph(top, hops, g["edge_index"], num_nodes=g["x"].size(0))
    E = g["n_fwd_edges"]
    return emask[:E] | emask[E:]


def path_logits(model, out, b, g, q, path, dev):
    """InfoNCE logits over [gold, candidates]: cos(W q, sum_steps mean(sigma(s_e) * rep_e)) / 0.1."""
    eids, step_of, path_of_step = [], [], []
    n_steps = 0
    for pi, pe in enumerate(path["edges"]):
        for es in pe:
            eids += es
            step_of += [n_steps] * len(es)
            path_of_step.append(pi)
            n_steps += 1
    e = torch.tensor(eids, device=dev)
    rep, lg = model.edge_repr(out, b, e, g)
    v = torch.sigmoid(lg).unsqueeze(-1) * rep
    so = torch.tensor(step_of, device=dev)
    step = torch.zeros(n_steps, v.size(1), device=dev).index_add_(0, so, v)
    cnt = torch.zeros(n_steps, device=dev).index_add_(0, so, torch.ones_like(lg)).clamp(min=1)
    step = step / cnt.unsqueeze(-1)
    P = torch.zeros(len(path["edges"]), v.size(1), device=dev).index_add_(
        0, torch.tensor(path_of_step, device=dev), step)
    P = F.normalize(P, dim=-1)
    qv = F.normalize(model.path_q(q[b]), dim=-1)
    return P @ qv / 0.1


def batch_loss(model, g, batch, qemb, args, dev, train=True):
    q = qemb[[s["q_idx"] for s in batch]].to(dev).float()
    out = model(g, q)
    N, E = g["x"].size(0), g["n_fwd_edges"]
    losses = {"node": 0.0, "edge": 0.0, "path": 0.0, "ans": 0.0}
    total = 0.0
    for b, s in enumerate(batch):
        w = s["weight"]
        yn = torch.zeros(N, device=dev); yn[s["nodes"]] = 1
        ya = torch.zeros(N, device=dev); ya[s["ans"]] = 1
        ln = bce_pw(out["node_logit"][b], yn)
        la = bce_pw(out["ans_logit"][b], ya)
        # edges: region (3 hops around top-64 nodes) + gold edges, negatives subsampled
        with torch.no_grad():
            reg = edge_region(out["node_logit"][b].detach(), g)
        ye = torch.zeros(E, dtype=torch.bool, device=dev)
        if s["edges"]:
            ye[s["edges"]] = True
        neg = (reg & ~ye).nonzero().squeeze(-1)
        if neg.numel() > args.max_neg_edges:
            neg = neg[torch.randperm(neg.numel(), device=dev)[: args.max_neg_edges]]
        pos = ye.nonzero().squeeze(-1)
        if pos.numel() > args.max_neg_edges:
            pos = pos[torch.randperm(pos.numel(), device=dev)[: args.max_neg_edges]]
        eids = torch.cat([pos, neg])
        le = torch.zeros((), device=dev)
        if eids.numel() and pos.numel():
            _, lg = model.edge_repr(out, b, eids, g)
            tgt = torch.cat([torch.ones(pos.numel(), device=dev), torch.zeros(neg.numel(), device=dev)])
            le = bce_pw(lg, tgt)
        lp = torch.zeros((), device=dev)
        if args.lambda_path > 0 and s["path"] is not None and len(s["path"]["paths"]) > 1:
            pth = s["path"]
            if train:
                k = min(len(pth["paths"]) - 1, args.n_siblings)
                pth = {"paths": pth["paths"][: k + 1], "edges": pth["edges"][: k + 1]}
            lgts = path_logits(model, out, b, g, q, pth, dev)
            lp = F.cross_entropy(lgts.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
        l = ln + le + args.lambda_path * lp + 0.5 * la
        total = total + w * l
        for k_, v in zip(losses, (ln, le, lp, la)):
            losses[k_] += float(v.detach() if torch.is_tensor(v) else v) / len(batch)
    return total / len(batch), losses, out


@torch.no_grad()
def evaluate(model, graphs, qemb, args, dev, max_q=None):
    """Gold-path Hit@{1,5,10} over [gold + up to 63 candidates]; node recall@10/50 (capped)."""
    model.eval()
    hits = {1: [], 5: [], 10: []}
    nrec = {10: [], 50: []}
    for g0 in graphs:
        g = to_dev(g0, dev)
        ss = g0["samples"][:max_q] if max_q else g0["samples"]
        for i in range(0, len(ss), args.batch):
            batch = ss[i:i + args.batch]
            q = qemb[[s["q_idx"] for s in batch]].to(dev).float()
            out = model(g, q)
            for b, s in enumerate(batch):
                top = out["node_logit"][b].topk(min(50, g["x"].size(0))).indices.tolist()
                Y = set(s["nodes"])
                for k in nrec:
                    nrec[k].append(len(Y & set(top[:k])) / min(k, len(Y)))
                if s["path"] is not None and len(s["path"]["paths"]) > 1:
                    lg = path_logits(model, out, b, g, q, s["path"], dev)
                    rank = int((lg > lg[0]).sum()) + 1
                    for k in hits:
                        hits[k].append(float(rank <= k))
    model.train()
    r = {f"path_hit@{k}": float(np.mean(v)) if v else 0.0 for k, v in hits.items()}
    r.update({f"node_recall@{k}": float(np.mean(v)) for k, v in nrec.items()})
    r["n_path_q"] = len(hits[10])
    return r


def gcs_mirror(path):
    bucket = os.environ.get("BRIDGE_BUCKET")
    if bucket:
        subprocess.run(["gsutil", "-q", "cp", str(path), f"{bucket}/ckpt/"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
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
    ap.add_argument("--buildings-limit", type=int)
    ap.add_argument("--val-max-q", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.no_path_loss:
        args.lambda_path = 0.0
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = kg.split()
    train = load_split(sp["train"], args.buildings_limit)
    val = load_split(sp["val"], args.buildings_limit)
    qemb = torch.load(PREP / "qemb.pt", weights_only=False)["emb"]
    model = BRIDGE(d=args.d, rel_layers=args.rel_layers, ent_layers=args.ent_layers, dropout=args.dropout,
                   hard_seeding=args.hard_seeding, relation_graph=not args.no_relation_graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    CKPT.mkdir(parents=True, exist_ok=True)
    logf = open(CKPT / f"{args.name}.jsonl", "a")
    cfg = vars(args) | {"init": "scratch (no ULTRA checkpoint)", "n_train_bldg": len(train), "n_val_bldg": len(val)}
    logf.write(json.dumps({"config": cfg}) + "\n"); logf.flush()
    # batches: 16 questions from the same building
    batches = []
    for gi, g in enumerate(train):
        ss = g["samples"]
        for i in range(0, len(ss), args.batch):
            batches.append((gi, i))
    best, bad, t0 = -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        rng = random.Random(args.seed + ep)
        for g in train:
            rng.shuffle(g["samples"])
        rng.shuffle(batches)
        agg, n = {"node": 0, "edge": 0, "path": 0, "ans": 0, "total": 0}, 0
        cache = {}
        for step, (gi, i) in enumerate(batches):
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
        ev = evaluate(model, val, qemb, args, dev, args.val_max_q)
        rec = {"epoch": ep, "train": tr, "val": ev, "elapsed_s": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        score = ev["path_hit@10"] + 1e-3 * ev["node_recall@50"]
        if score > best:
            best, bad = score, 0
            path = CKPT / f"{args.name}.pt"
            torch.save({"state": model.state_dict(), "config": cfg, "epoch": ep, "val": ev}, path)
            gcs_mirror(path)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    logf.write(json.dumps({"done": True, "best_score": best, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    logf.close()
    gcs_mirror(CKPT / f"{args.name}.jsonl")


if __name__ == "__main__":
    main()
