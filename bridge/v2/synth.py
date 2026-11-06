"""Synthetic multi-column SELECT queries over Build2SPARQL TRAIN / VAL buildings (never test).

mine:  class-level schema per building (most-specific asserted ABox types) -> random walks of
       1-4 hops (223P semantic relations and Brick relations up-weighted, 223P connection
       topology down-weighted), optional second branch from an anchor, plus 1-3 attributes of
       path entities: a literal (rdfs:label, ref:hasTimeseriesId, BACnet object-name, ...), an
       untyped IRI value (qudt:hasUnit, hasQuantityKind, brick:hasUnit, ...) or one hop through an
       untyped / external-reference node (ref:hasExternalReference/ref:hasTimeseriesId, ...).
       3-6 projected variables; validated by execution (1..2000 DISTINCT rows); deduped by a
       class-level signature; labelled with bridge.v2.labeler (same as labels_v2).
       -> runs/v2/synth_raw/{b}.json
para:  template question + Gemini 2.5 Flash paraphrases (technical, plain), 8 queries / request,
       <= 4 concurrent, 429 backoff, usage -> runs/v2/gemini_usage.jsonl, cache -> runs/v2/para_cache.jsonl
       -> runs/v2/synth/{b}.json  (questions = [template, technical, plain])

python -m bridge.v2.synth mine [--buildings ...] [--split train val] [--procs 8]
python -m bridge.v2.synth para [--buildings ...] [--split train val] [--budget 5.0] [--no-gemini]
"""
import argparse
import collections
import hashlib
import json
import math
import random
import re
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pyoxigraph as ox

from bridge import kg
from bridge.graph_build import split_name
from bridge.labels import run_with_timeout
from bridge.v2 import common as C
from bridge.v2.labeler import Ctx, label_query

MAX_ROWS = 2000
QTIMEOUT = 20
TARGET = {"s223": 35, "default": 10}
PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "brick": "https://brickschema.org/schema/Brick#",
    "ref": "https://brickschema.org/schema/Brick/ref#",
    "s223": "http://data.ashrae.org/standard223#",
    "qudt": "http://qudt.org/schema/qudt/",
    "bacnet": "http://data.ashrae.org/bacnet/2020#",
    "bgm": "https://buildgraph.org/meta#",
}
PRIO = {"hasDomainSpace": 4, "contains": 3, "hasProperty": 4, "hasExternalReference": 4, "observes": 4,
        "hasObservationLocation": 4, "hasPhysicalLocation": 3, "hasDomain": 2, "encloses": 3, "mapsTo": 2,
        "actuates": 3, "actuatedByProperty": 3, "hasRole": 1, "connectedTo": 2, "connectedFrom": 2,
        "hasPoint": 3, "isPointOf": 3, "feeds": 3, "isFedBy": 3, "hasLocation": 3, "isLocationOf": 3,
        "hasPart": 3, "isPartOf": 3, "controls": 2, "hasUnit": 3, "hasQuantityKind": 3}
TOPO = {"cnx", "hasConnectionPoint", "isConnectionPointOf", "connectsAt", "connectsThrough",
        "connectedThrough", "connected", "connectsFrom", "connectsTo", "hasMedium"}
SKIP_P = {kg.RDF_TYPE, kg.SUBCLASS}


def pw(p):
    ln = kg.local_name(p)
    return 0.15 if ln in TOPO else PRIO.get(ln, 1.0)


def qname(iri):
    for k, ns in PREFIXES.items():
        if iri.startswith(ns) and re.fullmatch(r"[A-Za-z_][\w\-]*", iri[len(ns):]):
            return f"{k}:{iri[len(ns):]}"
    return f"<{iri}>"


def camel(s):
    w = re.sub(r"[^0-9A-Za-z]+", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)).split()
    if not w:
        return "x"
    out = w[0].lower() + "".join(x[:1].upper() + x[1:].lower() for x in w[1:])
    return out if out[0].isalpha() else "v" + out


def attr_word(path):
    ln = kg.local_name(path[-1])
    ln = re.sub(r"^has(?=[A-Z])", "", ln)
    return split_name(ln)


# ----------------------------------------------------------------------------- schema
def schema(ctx):
    prim = {s: c for s, c in ctx.prim.items() if c and not c.startswith(kg.OWL)}
    members = collections.defaultdict(list)
    for s, c in prim.items():
        members[c].append(s)
    steps = collections.Counter()
    out_by = collections.defaultdict(list)
    for t in ctx.abox:
        s, p, o = t.subject.value, t.predicate.value, t.object
        if p in SKIP_P or p.startswith(kg.OWL) or s.startswith(kg.OWL):
            continue
        out_by[s].append((p, o))
        if isinstance(o, ox.NamedNode) and s in prim and o.value in prim:
            steps[(prim[s], p, prim[o.value])] += 1
    adj = collections.defaultdict(list)
    for (a, p, b), n in steps.items():
        if a != b:
            adj[a].append((p, 1, b, n))
            adj[b].append((p, -1, a, n))
    # attributes per class
    attrs = {}
    for c, ms in members.items():
        cnt = collections.Counter()
        kinds = {}
        for s in ms:
            seen = set()
            for p, o in out_by.get(s, ()):
                if isinstance(o, ox.Literal):
                    if C.attr_id(s, p, o.value) in ctx.attr_ok:
                        seen.add(((p,), "lit"))
                elif isinstance(o, ox.NamedNode) and (o.value not in prim or "ExternalReference" in p):
                    m = o.value
                    if not m.startswith("urn:bnode:") and m not in ctx.types:
                        seen.add(((p,), "iri"))
                    via = m not in ctx.types or "ExternalReference" in p
                    if via:
                        for p2, o2 in out_by.get(m, ()):
                            if isinstance(o2, ox.Literal):
                                if C.attr_id(m, p2, o2.value) in ctx.attr_ok:
                                    seen.add(((p, p2), "lit"))
                            elif (isinstance(o2, ox.NamedNode) and not o2.value.startswith("urn:bnode:")
                                  and o2.value not in ctx.types and o2.value not in prim):
                                seen.add(((p, p2), "iri"))
            for k in sorted(seen):
                cnt[k[0]] += 1
                kinds[k[0]] = k[1]
        attrs[c] = [{"path": k, "kind": kinds[k], "cov": n / len(ms)} for k, n in cnt.items()]
    return members, adj, attrs


def attr_weight(a):
    ln = kg.local_name(a["path"][-1])
    w = 0.6 if a["path"][-1] == kg.LABEL else 1.5
    if ln in ("hasTimeseriesId", "hasUnit", "hasQuantityKind", "object-name", "object-identifier",
              "device-identifier", "hasValue", "storedAt", "hasExternalReference", "description") \
            or len(a["path"]) == 2:
        w = 3.0
    if ln in ("hasMedium", "hasRole", "hasEnumerationKind", "ofMedium", "hasAspect"):
        w = 0.8
    return w * (0.3 + a["cov"])


# ----------------------------------------------------------------------------- generation
def gen_query(rng, members, adj, attrs):
    cands = [c for c in members if adj.get(c) or attrs.get(c)]
    if not cands:
        return None
    def rich(c):
        return sum(attr_weight(a) for a in attrs.get(c, []) if a["path"][-1] != kg.LABEL)

    wts = [math.sqrt(len(members[c])) * (1 + max((pw(p) for p, *_ in adj.get(c, [])), default=0))
           * (1 + min(rich(c), 6.0)) for c in cands]
    c0 = rng.choices(cands, wts)[0]
    vars_ = [c0]                      # var index -> class
    edges = []                        # (i, p, j) forward meaning: var_i -p-> var_j
    L = rng.choice([0, 1, 1, 2, 2, 3, 3, 4])

    def extend(start, n):
        cur = start
        for _ in range(n):
            opts = [(p, d, b, k) for p, d, b, k in adj.get(vars_[cur], []) if b not in vars_]
            if not opts:
                return
            p, d, b, _ = rng.choices(opts, [pw(p) * math.log(2 + k) for p, d, b, k in opts])[0]
            vars_.append(b)
            j = len(vars_) - 1
            edges.append((cur, p, j) if d == 1 else (j, p, cur))
            cur = j

    branched = False
    if rng.random() < 0.35:
        # two branches from one anchor (var 0), or a side branch off an internal node of a 2-hop chain
        l1 = rng.choice([1, 2])
        extend(0, l1)
        n0 = len(vars_)
        anchor = 1 if (len(vars_) == 3 and rng.random() < 0.5) else 0
        if n0 >= 2:
            extend(anchor, rng.choice([1, 1, 2]))
        branched = len(vars_) > n0
    else:
        extend(0, L)
    # attributes
    n_attr = rng.choice([1, 1, 2, 2, 3])
    owners = [i for i, c in enumerate(vars_) if attrs.get(c)]
    chosen = []
    for _ in range(n_attr * 3):
        if len(chosen) >= n_attr or not owners:
            break
        i = rng.choices(owners, [0.5 + min(rich(vars_[o]), 6.0) for o in owners])[0]
        opts = [a for a in attrs[vars_[i]] if (i, a["path"]) not in {(x, y["path"]) for x, y in chosen}]
        if not opts:
            continue
        chosen.append((i, rng.choices(opts, [attr_weight(a) for a in opts])[0]))
    n_ent = len(vars_)
    if not chosen and n_ent < 3:
        return None
    arity = rng.choice([3, 3, 4, 4, 5, 6])
    n_ent_proj = max(1, min(n_ent, arity - len(chosen)))
    if n_ent_proj + len(chosen) < 3:
        return None
    ent_proj = sorted(rng.sample(range(n_ent), n_ent_proj))
    return {"vars": vars_, "edges": edges, "attrs": chosen, "ent_proj": ent_proj, "branched": branched}


def signature(q):
    v = q["vars"]
    e = sorted((v[i], p, v[j]) for i, p, j in q["edges"])
    a = sorted((v[i], a["path"]) for i, a in q["attrs"])
    pr = sorted(v[i] for i in q["ent_proj"])
    return hashlib.sha1(json.dumps([e, a, pr], sort_keys=True).encode()).hexdigest()[:16]


def render(q):
    vars_ = q["vars"]
    names, used = [], collections.Counter()
    for c in vars_:
        n = camel(kg.local_name(c))
        used[n] += 1
        names.append(n if used[n] == 1 else f"{n}{used[n]}")
    body, proj, roles = [], [], []
    for i, c in enumerate(vars_):
        body.append(f"    ?{names[i]} rdf:type {qname(c)} .")
    for i, p, j in q["edges"]:
        body.append(f"    ?{names[i]} {qname(p)} ?{names[j]} .")
    attr_vars = []
    taken = set(names)
    for i, a in q["attrs"]:
        base = names[i] + "".join(w.capitalize() for w in attr_word(a["path"]).split()) or names[i] + "Value"
        av = base
        k = 2
        while av in taken:
            av, k = f"{base}{k}", k + 1
        taken.add(av)
        if len(a["path"]) == 1:
            body.append(f"    ?{names[i]} {qname(a['path'][0])} ?{av} .")
        else:
            mid = av + "Node"
            taken.add(mid)
            body.append(f"    ?{names[i]} {qname(a['path'][0])} ?{mid} .")
            body.append(f"    ?{mid} {qname(a['path'][1])} ?{av} .")
        attr_vars.append((av, i, a))
    for i in q["ent_proj"]:
        proj.append(names[i])
        roles.append({"var": names[i], "kind": "entity", "class": vars_[i]})
    for av, i, a in attr_vars:
        proj.append(av)
        roles.append({"var": av, "kind": a["kind"], "owner": names[i], "owner_class": vars_[i],
                      "path": list(a["path"])})
    btxt = "\n".join(body)
    pre = "\n".join(f"PREFIX {k}: <{v}>" for k, v in PREFIXES.items() if re.search(rf"(?<![\w?]){k}:", btxt))
    sparql = f"{pre}\n\nSELECT DISTINCT {' '.join('?' + x for x in proj)} WHERE {{\n" + "\n".join(body) + "\n}"
    return sparql, names, roles


def template_question(q, names, roles):
    vars_ = q["vars"]
    cn = [split_name(kg.local_name(c)) for c in vars_]
    # clauses: describe edges as relative clauses starting from var 0 (BFS order)
    clauses = []
    for i, p, j in q["edges"]:
        clauses.append(f"a {cn[i]} {split_name(kg.local_name(p))} a {cn[j]}")
    descs = []
    for r in roles:
        if r["kind"] == "entity":
            descs.append(f"the {split_name(kg.local_name(r['class']))}")
        else:
            descs.append(f"the {attr_word(r['path'])} of the {split_name(kg.local_name(r['owner_class']))}")
    lst = ", ".join(descs[:-1]) + f" and {descs[-1]}" if len(descs) > 1 else descs[0]
    if clauses:
        return f"Where {' and '.join(clauses)}, list {lst}."
    return f"For each {cn[0]}, list {lst}."


def mine_building(b, target=None, max_attempts=None):
    t0 = time.time()
    ctx = Ctx(b)
    target = target or TARGET.get(ctx.stratum, TARGET["default"])
    members, adj, attrs = schema(ctx)
    rng = random.Random(zlib.crc32(b.encode()))
    seen, out, stats = set(), [], collections.Counter()
    max_attempts = max_attempts or 40 * target
    for _ in range(max_attempts):
        if len(out) >= target:
            break
        q = gen_query(rng, members, adj, attrs)
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
        # every projected column must be represented by at least one node
        if any(not rec["Y_ans_cols"].get(r["var"]) for r in roles):
            stats["col_unmapped"] += 1
            continue
        stats["ok"] += 1
        has_lit = any(r["kind"] == "lit" for r in roles)
        out.append({"query_id": f"SYN_{len(out) + 1:03d}", "signature": sig, "source": "synth",
                    "pattern_type": "synth_branch" if q["branched"] else ("synth_chain" if q["edges"] else "synth_attr"),
                    "sparql_query": sparql, "roles": roles, "n_distinct_rows": len(rows),
                    "has_literal_proj": has_lit, "n_hops": len(q["edges"]),
                    "template": template_question(q, names, roles), "weight": 1.0, **rec})
    return {"building": b, "stratum": ctx.stratum, "split": C.split_of(b), "graph_variant": C.graph_variant(b),
            "n_classes": len(members), "stats": dict(stats), "seconds": round(time.time() - t0, 1), "queries": out}


def _mine_job(b):
    try:
        return mine_building(b)
    except Exception as e:  # noqa: BLE001
        return {"building": b, "fail": f"{type(e).__name__}: {e}"}


def main_mine(buildings, procs, force=False):
    C.SRAW.mkdir(parents=True, exist_ok=True)
    todo = [b for b in buildings if force or not (C.SRAW / f"{b}.json").exists()]
    with ProcessPoolExecutor(procs) as ex:
        futs = {ex.submit(_mine_job, b): b for b in todo}
        for f in as_completed(futs):
            r = f.result()
            b = r["building"]
            if "fail" in r:
                print(f"{b}: FAIL {r['fail']}", flush=True)
                continue
            (C.SRAW / f"{b}.json").write_text(json.dumps(r))
            nl = sum(q["has_literal_proj"] for q in r["queries"])
            print(f"{b} [{r['stratum']}/{r['split']}]: {len(r['queries'])} queries (lit {nl}) {r['stats']} {r['seconds']}s", flush=True)


# ----------------------------------------------------------------------------- paraphrasing
PRICE_IN, PRICE_OUT = 0.15e-6, 0.60e-6
_LOCK = threading.Lock()
_SPENT = {"usd": 0.0, "in": 0, "out": 0, "calls": 0}

PROMPT = """You write natural-language questions that building engineers or facility managers would ask about a building knowledge graph (Brick Schema / ASHRAE 223P).
For each item below you get a template question and the SPARQL query it describes. Write TWO paraphrases per item:
- "technical": precise building-automation / HVAC wording (may use domain terms such as sensor, VAV box, zone, timeseries id, unit, external reference); do not use SPARQL, variable names, prefixes like "s223:" or camelCase identifiers.
- "plain": everyday wording a facility manager would type; no ontology jargon.
Both paraphrases must be a single question or request that asks for ALL the same returned columns (each listed item) and keeps the same constraints/relationships. Do not invent filters or specific entity names.
Return ONLY a JSON list: [{{"id": <id>, "technical": "...", "plain": "..."}}, ...]

Items:
{items}"""


def _usage_log(rec):
    with _LOCK:
        with open(C.V2 / "gemini_usage.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def gemini_call(model, prompt, budget, max_tries=7):
    from google.api_core import exceptions as gex
    delay = 5.0
    for attempt in range(max_tries):
        with _LOCK:
            if _SPENT["usd"] >= budget:
                raise RuntimeError("budget exhausted")
        try:
            r = model.generate_content(prompt, generation_config={"temperature": 0.7,
                                                                  "response_mime_type": "application/json"})
            um = r.usage_metadata
            pin = int(getattr(um, "prompt_token_count", 0) or 0)
            tot = int(getattr(um, "total_token_count", 0) or 0)
            pout = max(0, tot - pin)
            usd = pin * PRICE_IN + pout * PRICE_OUT
            with _LOCK:
                _SPENT["usd"] += usd; _SPENT["in"] += pin; _SPENT["out"] += pout; _SPENT["calls"] += 1
            _usage_log({"t": time.time(), "prompt_tokens": pin, "output_tokens_incl_thinking": pout,
                        "usd": round(usd, 6), "cum_usd": round(_SPENT["usd"], 4)})
            try:
                return r.text
            except Exception:  # noqa: BLE001
                return ""
        except (gex.ResourceExhausted, gex.ServiceUnavailable, gex.InternalServerError, gex.DeadlineExceeded) as e:
            time.sleep(delay)
            delay = min(delay * 2, 160)
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "Resource exhausted" in str(e):
                time.sleep(delay)
                delay = min(delay * 2, 160)
            else:
                print("gemini error:", type(e).__name__, str(e)[:200], flush=True)
                time.sleep(2)
    return ""


def _parse_json_list(txt):
    try:
        v = json.loads(txt)
    except Exception:  # noqa: BLE001
        m = re.search(r"\[.*\]", txt or "", re.S)
        if not m:
            return []
        try:
            v = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
    return v if isinstance(v, list) else []


def paraphrase_all(items, budget, workers=4, batch=8):
    """items: list of (key, template, sparql). Returns {key: {"technical":..., "plain":...}} (cached)."""
    cache_f = C.V2 / "para_cache.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["key"]] = d
    todo = [it for it in items if it[0] not in cache]
    if not todo:
        return cache
    import vertexai
    from vertexai.generative_models import GenerativeModel
    import os
    vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"],
                  location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    model = GenerativeModel("gemini-2.5-flash")
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def run(chunk):
        parts = []
        for k, (key, t, s) in enumerate(chunk):
            s_body = re.sub(r"PREFIX[^\n]*\n", "", s).strip()
            parts.append(f"id: {k}\ntemplate: {t}\nsparql:\n{s_body}")
        body = "\n\n".join(parts)
        res = _parse_json_list(gemini_call(model, PROMPT.format(items=body), budget))
        got = {}
        for d in res:
            try:
                k = int(d["id"])
                if 0 <= k < len(chunk) and d.get("technical") and d.get("plain"):
                    got[chunk[k][0]] = {"key": chunk[k][0], "technical": str(d["technical"]).strip(),
                                        "plain": str(d["plain"]).strip()}
            except Exception:  # noqa: BLE001
                continue
        return got

    done = 0
    with ThreadPoolExecutor(workers) as ex, open(cache_f, "a", encoding="utf-8") as cf:
        futs = [ex.submit(run, ch) for ch in chunks]
        for f in as_completed(futs):
            try:
                got = f.result()
            except Exception as e:  # noqa: BLE001
                print("chunk failed:", e, flush=True)
                continue
            for k, d in got.items():
                cache[k] = d
                cf.write(json.dumps(d) + "\n")
            cf.flush()
            done += 1
            if done % 20 == 0:
                print(f"  paraphrase {done}/{len(chunks)} chunks, ${_SPENT['usd']:.3f}", flush=True)
    return cache


def main_para(buildings, budget, use_gemini=True):
    C.SDIR.mkdir(parents=True, exist_ok=True)
    raws = {}
    for b in buildings:
        f = C.SRAW / f"{b}.json"
        if f.exists():
            raws[b] = json.loads(f.read_text())
    items = [(f"{b}|{q['signature']}", q["template"], q["sparql_query"]) for b, r in raws.items() for q in r["queries"]]
    para = paraphrase_all(items, budget) if use_gemini and items else {}
    n_para = 0
    for b, r in raws.items():
        for q in r["queries"]:
            p = para.get(f"{b}|{q['signature']}")
            q["questions"] = [q["template"]] + ([p["technical"], p["plain"]] if p else [])
            q["sources"] = ["template"] + (["gemini_technical", "gemini_plain"] if p else [])
            n_para += bool(p)
        (C.SDIR / f"{b}.json").write_text(json.dumps(r))
    print(f"wrote {len(raws)} synth files; {n_para}/{len(items)} paraphrased; this run "
          f"${_SPENT['usd']:.4f} ({_SPENT['in']} in / {_SPENT['out']} out tokens, {_SPENT['calls']} calls)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "para"])
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sp = kg.split()
    assert not set(a.split) - {"train", "val"}, "synthetic data only for train/val buildings"
    allowed = set(sp["train"]) | set(sp["val"])
    bs = a.buildings or [b for k in a.split for b in sp[k]]
    bad = [b for b in bs if b not in allowed]
    assert not bad, f"not train/val buildings: {bad}"
    if a.cmd == "mine":
        main_mine(bs, a.procs, a.force)
    else:
        main_para(bs, a.budget, not a.no_gemini)
