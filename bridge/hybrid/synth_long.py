"""Long-path synthetic SELECT queries (3-7 hops, literal tails) on CLEANED Build2SPARQL graphs.

Extends bridge.v2.synth (same class-level schema, relation weights, rendering, labelling):
  * a target total hop count T in [3, 7] is sampled first (hops = all non-type triple patterns,
    literal / external-reference tails included, as in the BuildingQA gold query graphs);
  * shape: entity CHAIN, or TWO-BRANCH STAR around an anchor class;
  * 1-3 attribute tails on path entities (preferably the far ends of the branches): direct literal
    predicates (rdfs:label, BACnet object-name, ...), or two hops through an untyped / external-
    reference node (ref:hasExternalReference/ref:hasTimeseriesId, .../ref:ifcName, .../endpoint, ...);
  * 2-6 projected columns mixing entity and literal columns (>= 1 literal column required);
  * validated by execution (1..2000 DISTINCT rows), deduped by class-level signature, labelled with
    bridge.v2.labeler (Y_V / Y_E / Y_ans / Y_ans_cols / P_ast / chain / P_inst) on the hybrid graph.
TRAIN and VAL buildings only (separate files carry the split).

mine:  python -m bridge.hybrid.synth_long mine [--split train val] [--buildings ...] [--procs 8]
       -> runs/hybrid/synth_long_raw/{b}.json   (questions = [template])
para:  python -m bridge.hybrid.synth_long para [--yes-call-gemini] [--budget 3.0]
       DRY RUN unless --yes-call-gemini: prints #items / chunks / cost estimate and exits.
       With --no-gemini writes template-only files. Paraphrases via bridge.v2.synth.paraphrase_all
       (same prompt, Gemini 2.5 Flash, same cache format) with cache runs/hybrid/para_cache.jsonl and
       usage log runs/hybrid/gemini_usage.jsonl  -> runs/hybrid/synth_long/{b}.json
"""
import argparse
import collections
import json
import math
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from bridge import kg
from bridge.hybrid import common as H
from bridge.labels import run_with_timeout
from bridge.v2 import common as C2
from bridge.v2.labeler import Ctx, label_query
from bridge.v2.synth import (MAX_ROWS, QTIMEOUT, attr_weight, pw, render, schema, signature, template_question)

TARGET = {"s223": 30, "default": 15}
HOP_W_TRAIN = {3: 1.0, 4: 1.2, 5: 1.2, 6: 1.0, 7: 0.8}


def hybrid_ctx(b):
    return Ctx(b, nt=H.clean_nt(b), strat=C2.stratum(b))


# ----------------------------------------------------------------------------- generation
def gen_long(rng, members, adj, attrs, hop_w, require_lit=True, star_p=0.4):
    cands = [c for c in members if adj.get(c)]
    if not cands:
        return None

    def rich(c):
        return sum(attr_weight(a) for a in attrs.get(c, []) if a["path"][-1] != kg.LABEL)

    wts = [math.sqrt(len(members[c])) * (1 + max((pw(p) for p, *_ in adj.get(c, [])), default=0))
           * (1 + min(rich(c), 6.0)) for c in cands]
    T = rng.choices(list(hop_w), list(hop_w.values()))[0]
    plan = [2 if rng.random() < 0.45 else 1 for _ in range(rng.choice([1, 1, 2, 2, 3]))]
    E = max(1, T - sum(plan))
    c0 = rng.choices(cands, wts)[0]
    vars_, edges = [c0], []

    def extend(start, n):
        cur = start
        for _ in range(n):
            opts = [(p, d, b, k) for p, d, b, k in adj.get(vars_[cur], []) if b not in vars_]
            if not opts:
                return cur
            p, d, b, _ = rng.choices(opts, [pw(p) * math.log(2 + k) for p, d, b, k in opts])[0]
            vars_.append(b)
            j = len(vars_) - 1
            edges.append((cur, p, j) if d == 1 else (j, p, cur))
            cur = j
        return cur

    ends = []
    star = E >= 2 and rng.random() < star_p
    if star:
        l1 = rng.randint(1, E - 1)
        ends.append(extend(0, l1))
        n0 = len(vars_)
        ends.append(extend(0, E - l1))
        if len(vars_) == n0:
            star = False
    else:
        ends.append(extend(0, E))
        ends.append(0)
    if not edges:
        return None
    # attributes, preferably on branch ends (long spans), planned lengths first
    owners_pref = list(dict.fromkeys(ends + list(range(len(vars_)))))
    chosen = []
    for want in plan:
        for _ in range(6):
            pool = [i for i in owners_pref if attrs.get(vars_[i])]
            if not pool:
                break
            i = rng.choices(pool, [3.0 if k in ends else 0.7 for k in pool])[0]
            taken = {(x, tuple(y["path"])) for x, y in chosen}
            opts = [a for a in attrs[vars_[i]] if (i, tuple(a["path"])) not in taken]
            if not opts:
                continue
            same = [a for a in opts if len(a["path"]) == want]
            opts = same or opts
            chosen.append((i, rng.choices(opts, [attr_weight(a) for a in opts])[0]))
            break
    if require_lit and not any(a["kind"] == "lit" for _, a in chosen):
        return None
    total = len(edges) + sum(len(a["path"]) for _, a in chosen)
    if total < min(hop_w) or total > max(hop_w):
        return None
    n_cols = rng.choice([2, 3, 3, 4, 4, 5, 6])
    n_ent = max(0 if rng.random() < 0.15 else 1, min(len(vars_), n_cols - len(chosen)))
    if n_ent + len(chosen) < 2 or n_ent + len(chosen) > 6:
        return None
    pref = [e for e in dict.fromkeys(ends) if e < len(vars_)]
    rest = [i for i in range(len(vars_)) if i not in pref]
    rng.shuffle(rest)
    ent_proj = sorted((pref + rest)[:n_ent])
    return {"vars": vars_, "edges": edges, "attrs": chosen, "ent_proj": ent_proj, "branched": star}


def query_span(q):
    """Number of hops and the longest simple path (tree diameter) of the query graph."""
    adj = collections.defaultdict(set)
    for i, _, j in q["edges"]:
        adj[("v", i)].add(("v", j)); adj[("v", j)].add(("v", i))
    for k, (i, a) in enumerate(q["attrs"]):
        prev = ("v", i)
        for h in range(len(a["path"])):
            nxt = ("a", k, h)
            adj[prev].add(nxt); adj[nxt].add(prev)
            prev = nxt

    def far(src):
        dist, stack = {src: 0}, [src]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    stack.append(w)
        return max(dist.items(), key=lambda kv: kv[1])

    if not adj:
        return 0
    u, _ = far(next(iter(adj)))
    return far(u)[1]


def mine(ctx, target, hop_w, seed, require_lit=True, exclude_sigs=(), max_attempts=None, prefix="SYNL"):
    """Hop-stratified mining: per-hop quotas proportional to hop_w (long queries are rarer to find,
    so short valid ones are held in reserve and only used to fill up at the end)."""
    members, adj, attrs = schema(ctx)
    rng = random.Random(seed)
    seen, out, stats = set(exclude_sigs), [], collections.Counter()
    wsum = sum(hop_w.values())
    quota = {h: math.ceil(target * w / wsum) for h, w in hop_w.items()}
    reserve = []
    for _ in range(max_attempts or 200 * target):
        if len(out) >= target:
            break
        q = gen_long(rng, members, adj, attrs, hop_w, require_lit)
        if q is None:
            stats["gen_none"] += 1
            continue
        sig = signature(q)
        if sig in seen:
            stats["dup"] += 1
            continue
        seen.add(sig)
        sparql, names, roles = render(q)
        rows, err = run_with_timeout(ctx.store, sparql + f" LIMIT {MAX_ROWS + 1}", QTIMEOUT)
        if err or not rows or len(rows) > MAX_ROWS:
            stats["err" if err else ("empty" if not rows else "too_many")] += 1
            continue
        rec, why = label_query(ctx, sparql, QTIMEOUT)
        if rec is None or not rec["Y_ans"]:
            stats["label_" + (why or "noans")] += 1
            continue
        if any(not rec["Y_ans_cols"].get(r["var"]) for r in roles):
            stats["col_unmapped"] += 1
            continue
        n_hops = len(q["edges"]) + sum(len(a["path"]) for _, a in q["attrs"])
        dest = out if sum(x["n_hops"] == n_hops for x in out) < quota.get(n_hops, 0) else reserve
        stats["ok" if dest is out else "ok_reserve"] += 1
        dest.append({"query_id": "", "signature": sig, "source": "synth_long",
                    "pattern_type": "synthL_star" if q["branched"] else "synthL_chain",
                    "sparql_query": sparql, "roles": roles, "n_distinct_rows": len(rows),
                    "has_literal_proj": any(r["kind"] == "lit" for r in roles),
                    "n_lit_cols": sum(r["kind"] == "lit" for r in roles), "n_cols": len(roles),
                    "n_hops": n_hops, "n_entity_hops": len(q["edges"]), "span": query_span(q),
                    "template": template_question(q, names, roles), "weight": 1.0, **rec})
    rng.shuffle(reserve)
    reserve.sort(key=lambda x: -x["n_hops"])          # fill up with the longest leftovers first
    out += reserve[: max(0, target - len(out))]
    for k, x in enumerate(out, 1):
        x["query_id"] = f"{prefix}_{k:03d}"
    return out, stats


def mine_building(b):
    t0 = time.time()
    ctx = hybrid_ctx(b)
    target = TARGET.get(ctx.stratum, TARGET["default"])
    out, stats = mine(ctx, target, HOP_W_TRAIN, zlib.crc32(("long:" + b).encode()))
    for q in out:
        q["questions"], q["sources"] = [q["template"]], ["template"]
    return {"building": b, "stratum": ctx.stratum, "split": C2.split_of(b), "graph_variant": C2.graph_variant(b),
            "cleaned": True, "stats": dict(stats), "seconds": round(time.time() - t0, 1), "queries": out}


def _job(b):
    try:
        return mine_building(b)
    except Exception as e:  # noqa: BLE001
        return {"building": b, "fail": f"{type(e).__name__}: {e}"}


def summarize(recs, title):
    by = collections.defaultdict(list)
    for r in recs:
        by[r["stratum"]] += r["queries"]
    print(f"== {title}")
    for st, qs in sorted(by.items()):
        if not qs:
            continue
        hops = collections.Counter(q["n_hops"] for q in qs)
        span = collections.Counter(q["span"] for q in qs)
        lit = sum(q["n_lit_cols"] for q in qs) / max(1, sum(q["n_cols"] for q in qs))
        print(f"  {st}: {len(qs)} queries; hops {dict(sorted(hops.items()))}; span {dict(sorted(span.items()))}; "
              f"literal-column share {lit:.2f}; star {sum(q['pattern_type'] == 'synthL_star' for q in qs)}; "
              f"with P_inst {sum(bool(q['P_inst']) for q in qs)}", flush=True)


def main_mine(buildings, procs, force=False):
    H.SRAW.mkdir(parents=True, exist_ok=True)
    todo = [b for b in buildings if force or not (H.SRAW / f"{b}.json").exists()]
    with ProcessPoolExecutor(procs) as ex:
        futs = {ex.submit(_job, b): b for b in todo}
        for f in as_completed(futs):
            r = f.result()
            if "fail" in r:
                print(f"{r['building']}: FAIL {r['fail']}", flush=True)
                continue
            (H.SRAW / f"{r['building']}.json").write_text(json.dumps(r))
            print(f"{r['building']} [{r['stratum']}/{r['split']}]: {len(r['queries'])} {r['stats']} {r['seconds']}s", flush=True)
    recs = [json.loads((H.SRAW / f"{b}.json").read_text()) for b in buildings if (H.SRAW / f"{b}.json").exists()]
    for sp in ("train", "val"):
        summarize([r for r in recs if r["split"] == sp], f"long synth {sp}")


# ----------------------------------------------------------------------------- paraphrasing (NOT run here)
def main_para(buildings, budget, yes, use_gemini=True):
    raws = {b: json.loads((H.SRAW / f"{b}.json").read_text()) for b in buildings if (H.SRAW / f"{b}.json").exists()}
    items = [(f"L|{b}|{q['signature']}", q["template"], q["sparql_query"]) for b, r in raws.items() for q in r["queries"]]
    para = {}
    if use_gemini and items:
        cache_f = H.H / "para_cache.jsonl"
        have = set()
        if cache_f.exists():
            have = {json.loads(l)["key"] for l in cache_f.read_text(encoding="utf-8").splitlines() if l.strip()}
        todo = [it for it in items if it[0] not in have]
        n_chunks = math.ceil(len(todo) / 8)
        est = n_chunks * (3500 * 0.15e-6 + 2500 * 0.60e-6)
        print(f"{len(items)} queries, {len(todo)} not cached -> {n_chunks} Gemini 2.5 Flash requests, est. ${est:.2f}")
        if not yes:
            print("DRY RUN: no Gemini request made. Re-run with --yes-call-gemini to paraphrase "
                  "(or --no-gemini to write template-only files).")
            return
        from bridge.v2 import synth as S2
        C2.V2 = H.H          # redirect v2 cache / usage log to runs/hybrid (runtime only)
        para = S2.paraphrase_all(items, budget)
    H.SDIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for b, r in raws.items():
        for q in r["queries"]:
            p = para.get(f"L|{b}|{q['signature']}")
            q["questions"] = [q["template"]] + ([p["technical"], p["plain"]] if p else [])
            q["sources"] = ["template"] + (["gemini_technical", "gemini_plain"] if p else [])
            n += bool(p)
        (H.SDIR / f"{b}.json").write_text(json.dumps(r))
    print(f"wrote {len(raws)} files to {H.SDIR}; {n}/{len(items)} paraphrased")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "para"])
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--yes-call-gemini", action="store_true")
    ap.add_argument("--no-gemini", action="store_true")
    a = ap.parse_args()
    sp = kg.split()
    assert not set(a.split) - {"train", "val"}, "synthetic training data only for train/val buildings"
    allowed = set(sp["train"]) | set(sp["val"])
    bs = a.buildings or [b for k in a.split for b in sp[k]]
    bad = [b for b in bs if b not in allowed]
    assert not bad, f"not train/val buildings: {bad}"
    if a.cmd == "mine":
        main_mine(bs, a.procs, a.force)
    else:
        main_para(bs, a.budget, a.yes_call_gemini, not a.no_gemini)
