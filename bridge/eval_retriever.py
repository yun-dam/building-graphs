"""Step 6: retriever evaluation on the Build2SPARQL test split (33 buildings), per stratum.

Metrics (per question, then averaged):
  node recall@{10,50}, edge recall@{50,200}  -- |top-k ∩ Y| / min(k, |Y|)
  path hit@{1,5,10} -- rank of the gold class path among [gold + up to 63 alternatives sharing its
                       start (and preferably end) class]
  answer F1          -- BRIDGE: sigmoid(ans) > 0.5; baselines: top-10 ranked nodes
  context tokens / gold-step coverage -- on one register per query (rotating), 2k-token budget

python -m bridge.eval_retriever --models main abl_no_path ... [--split test] [--no-pcst]
"""
import argparse
import collections
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from bridge import kg
from bridge.baselines import KNNBaseline, PCSTBaseline
from bridge.data import path_key
from bridge.graph_build import embed
from bridge.retrieve import InfGraph, Prefixes, bridge_context, knn_context, load_model
from bridge.train import edge_region, path_logits

ROOT = kg.ROOT
PREP = ROOT / "runs" / "prep"


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def rec_at(order, gold, ks):
    gold = set(gold)
    if not gold:
        return {k: None for k in ks}
    order = order[: max(ks)]
    return {k: len(gold & set(order[:k])) / min(k, len(gold)) for k in ks}


def gold_steps(s):
    if not s["path"]:
        return None
    nodes, rels = s["path"]["paths"][0]
    return {(nodes[i], rels[i], nodes[i + 1]) for i in range(len(rels))}


def covered(steps_in_ctx, gsteps):
    if gsteps is None:
        return None
    inv = {(c, r ^ 1, a) for a, r, c in steps_in_ctx}
    return float(all(st in steps_in_ctx or st in inv for st in gsteps))


@torch.no_grad()
def eval_bridge(model, ig, samples, qemb, dev, do_ctx):
    res = []
    t = ig.t
    E = t["n_fwd_edges"]
    for i in range(0, len(samples), 16):
        batch = samples[i:i + 16]
        q = qemb[[s["q_idx"] for s in batch]].to(dev).float()
        out = model(t, q)
        for b, s in enumerate(batch):
            nl = out["node_logit"][b]
            norder = nl.argsort(descending=True).tolist()
            reg = edge_region(nl, t)
            eids = reg.nonzero().squeeze(-1)
            _, lg = model.edge_repr(out, b, eids, t) if eids.numel() else (None, torch.zeros(0, device=dev))
            eorder = eids[lg.argsort(descending=True)].tolist()
            r = {"nr": rec_at(norder, s["nodes"], (10, 50)), "er": rec_at(eorder, s["edges"], (50, 200))}
            ans = (torch.sigmoid(out["ans_logit"][b]) > 0.5).nonzero().squeeze(-1).tolist()
            r["ans_f1"] = f1(ans, s["ans"])
            if s["path"] and len(s["path"]["paths"]) > 1:
                pl = path_logits(model, out, b, ig.t, q, s["path"], dev)
                r["rank"] = int((pl > pl[0]).sum()) + 1
            res.append(r)
    if do_ctx:
        for j, s in enumerate(samples):
            if s["reg"] != (hash(s["qid"]) % 5) + 1:
                continue
            res[j]["ctx"] = True
    return res


def ctx_bridge(model, ig, question, s):
    block, info = bridge_context(model, ig, question)
    steps = set()
    for key in info["class_paths"]:
        for i in range(1, len(key), 2):
            steps.add((key[i - 1], key[i], key[i + 1]))
    return info["tokens"], covered(steps, gold_steps(s)), info["latency_s"]


def ctx_knn(knn, ig, question, s):
    block, info = knn_context(knn.idx, ig, question)
    pt = ig.g["primary_type"]
    ei, et = ig.g["edge_index"], ig.g["edge_type"]
    steps = {(int(pt[ei[0, e]]), int(et[e]), int(pt[ei[1, e]])) for e in info["edges"]}
    return info["tokens"], covered(steps, gold_steps(s)), info["latency_s"]


def path_texts(ig, paths):
    g = ig.g
    out = []
    for nodes, rels in paths:
        t = g["texts"][nodes[0]].split(".")[0]
        for i, r in enumerate(rels):
            t += f" {g['rel_texts'][r]} {g['texts'][nodes[i + 1]].split('.')[0]}"
        out.append(t)
    return out


def summarize(rows):
    def m(key, sub=None):
        v = [r[key][sub] if sub is not None else r[key] for r in rows
             if key in r and (r[key][sub] if sub is not None else r[key]) is not None]
        return float(np.mean(v)) if v else None
    ranks = [r["rank"] for r in rows if "rank" in r]
    out = {"n_questions": len(rows),
           "node_recall@10": m("nr", 10), "node_recall@50": m("nr", 50),
           "edge_recall@50": m("er", 50), "edge_recall@200": m("er", 200),
           "path_hit@1": float(np.mean([x <= 1 for x in ranks])) if ranks else None,
           "path_hit@5": float(np.mean([x <= 5 for x in ranks])) if ranks else None,
           "path_hit@10": float(np.mean([x <= 10 for x in ranks])) if ranks else None,
           "n_path_questions": len(ranks), "answer_f1": m("ans_f1")}
    ct = [r["tokens"] for r in rows if "tokens" in r]
    cv = [r["cov"] for r in rows if r.get("cov") is not None]
    lat = [r["lat"] for r in rows if "lat" in r]
    out["context_tokens"] = float(np.mean(ct)) if ct else None
    out["gold_steps_in_context"] = float(np.mean(cv)) if cv else None
    out["latency_s"] = float(np.mean(lat)) if lat else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["main"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-pcst", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "retriever_eval.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = kg.manifest()
    qall = torch.load(PREP / "qemb.pt", weights_only=False)
    qemb, qtexts = qall["emb"], qall["texts"]
    models = {}
    for name in a.models:
        f = ROOT / "runs" / "ckpt" / f"{name}.pt"
        if f.exists():
            models[name] = load_model(f, dev)[0]
    methods = list(models) + ["text-kNN"] + ([] if a.no_pcst else ["PCST"])
    rows = {m: collections.defaultdict(list) for m in methods}
    t0 = time.time()
    for b in kg.split()[a.split]:
        g = torch.load(ROOT / "runs" / "graphs" / f"{b}.pt", weights_only=False)
        prep = torch.load(PREP / f"{b}.pt", weights_only=False)
        g["rel_ids"] = prep["rel_ids"]
        ig = InfGraph(g, Prefixes(ROOT / man[b]["path"]), dev)
        samples = prep["samples"]
        strat = man[b]["stratum"]
        sel = lambda s: s["reg"] == (sum(map(ord, s["qid"])) % 5) + 1  # one register per query
        for name, model in models.items():
            res = eval_bridge(model, ig, samples, qemb, dev, False)
            for s, r in zip(samples, res):
                if sel(s) and name == a.models[0]:
                    r["tokens"], r["cov"], r["lat"] = ctx_bridge(model, ig, qtexts[s["q_idx"]], s)
                rows[name][strat].append(r)
        knn = KNNBaseline(ig)
        pcst = None if a.no_pcst else PCSTBaseline(ig, knn)
        for s in samples:
            q = qemb[s["q_idx"]].to(dev).float()
            eo, no, es = knn.rank(q)
            r = {"nr": rec_at(no.tolist(), s["nodes"], (10, 50)), "er": rec_at(eo.tolist(), s["edges"], (50, 200)),
                 "ans_f1": f1(no[:10].tolist(), s["ans"])}
            if s["path"] and len(s["path"]["paths"]) > 1:
                pe = embed(path_texts(ig, s["path"]["paths"])).to(dev)
                sc = pe @ q
                r["rank"] = int((sc > sc[0]).sum()) + 1
            if sel(s):
                r["tokens"], r["cov"], r["lat"] = ctx_knn(knn, ig, qtexts[s["q_idx"]], s)
            rows["text-kNN"][strat].append(r)
            if pcst is not None and sel(s):
                eo2, no2, nn_, ne_ = pcst.rank(q)
                rows["PCST"][strat].append({"nr": rec_at(no2.tolist(), s["nodes"], (10, 50)),
                                            "er": rec_at(eo2.tolist(), s["edges"], (50, 200)),
                                            "ans_f1": f1(no2[:10].tolist(), s["ans"]),
                                            "tokens": None, "sel_nodes": nn_, "sel_edges": ne_})
        print(f"{b} ({strat}) done {time.time()-t0:.0f}s", flush=True)
    summary = {}
    for m in methods:
        allr = [r for v in rows[m].values() for r in v]
        for r in allr:
            if r.get("tokens") is None:
                r.pop("tokens", None)
        summary[m] = {"all": summarize(allr), **{k: summarize(v) for k, v in rows[m].items()}}
    if "PCST" in summary:
        sel_n = [r["sel_nodes"] for v in rows["PCST"].values() for r in v]
        summary["PCST"]["all"]["mean_selected_nodes"] = float(np.mean(sel_n)) if sel_n else None
        summary["PCST"]["note"] = "evaluated on one register per query; path Hit@k n/a (PCST returns a subgraph, no path scores)"
    json.dump(summary, open(a.out, "w"), indent=1)
    for m in methods:
        s = summary[m]["all"]
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
