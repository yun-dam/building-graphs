"""Step 2: execute gold SPARQL, derive Y_V, Y_E, Y_ans, P* per (building, query).

Usage:
  python -m bridge.labels                # all split buildings
  python -m bridge.labels --buildings a b
  python -m bridge.labels --buildingqa   # BuildingQA gold answers (scoring only)
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import re
import threading
import time
from pathlib import Path

import pyoxigraph as ox
from rdflib import URIRef, Variable
from rdflib.paths import MulPath, SequencePath
from rdflib.plugins.sparql import prepareQuery

from bridge import kg

ROOT = kg.ROOT
B2S = ROOT / "data" / "Build2SPARQL_v1.0"
LDIR = ROOT / "runs" / "labels"
TIMEOUT = 60
ROW_CAP = 5000
MAX_Q = 60
FORCE = False
RDF_TYPE = URIRef(kg.RDF_TYPE)
SUBCLASS = URIRef(kg.SUBCLASS)


# ----------------------------------------------------------------------------- query AST
def _walk(node, out):
    if isinstance(node, dict):
        if node.get("triples") is not None and getattr(node, "name", "") == "BGP":
            out.extend(node["triples"])
        for v in node.values():
            _walk(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk(v, out)


def parse_patterns(sparql):
    """Return (edge_patterns, type_patterns, other_patterns, projected_vars)."""
    q = prepareQuery(sparql)
    triples = []
    _walk(q.algebra, triples)
    edges, types, other = [], [], []
    for s, p, o in triples:
        if isinstance(p, URIRef):
            if p == RDF_TYPE and isinstance(o, URIRef):
                types.append((s, o))
            else:
                edges.append((s, p, o))
        elif (isinstance(p, SequencePath) and len(p.args) == 2 and p.args[0] == RDF_TYPE
              and isinstance(p.args[1], MulPath) and p.args[1].path == SUBCLASS and isinstance(o, URIRef)):
            types.append((s, o))
        else:
            other.append((s, p, o))
    m = re.search(r"SELECT\s+(.+?)\s+WHERE", sparql, re.I | re.S)
    proj = re.findall(r"\?(\w+)", m.group(1)) if m else []
    # aggregation aliases are not bound in the body; keep only body vars
    body_vars = {str(t) for tr in triples for t in (tr[0], tr[2]) if isinstance(t, Variable)}
    proj = [v for v in dict.fromkeys(proj) if v in body_vars]
    return edges, types, other, proj


def select_star(sparql):
    """Rewrite to SELECT * over the same WHERE clause with a row cap."""
    m = re.search(r"\bSELECT\b", sparql, re.I)
    prefix = sparql[: m.start()]
    w = re.search(r"\bWHERE\s*\{", sparql[m.start():], re.I)
    start = m.start() + w.end() - 1
    depth = 0
    for i in range(start, len(sparql)):
        if sparql[i] == "{":
            depth += 1
        elif sparql[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return f"{prefix}SELECT * WHERE {sparql[start:end + 1]} LIMIT {ROW_CAP}"


def chain_order(edges):
    """If var-var edges form a simple path, return ordered steps [(v0,p,dir,v1),...]."""
    ve = [(str(s), str(p), str(o)) for s, p, o in edges if isinstance(s, Variable) and isinstance(o, Variable)]
    if not ve or len(ve) != len([e for e in edges if isinstance(e[0], Variable)]):
        return None
    adj = collections.defaultdict(list)
    for s, p, o in ve:
        adj[s].append((p, 1, o))
        adj[o].append((p, -1, s))
    if any(len(v) > 2 for v in adj.values()) or len(adj) != len(ve) + 1:
        return None
    ends = [v for v, n in adj.items() if len(n) == 1]
    if len(ends) != 2:
        return None
    # start at the end that has an outgoing edge first, for readability
    start = sorted(ends, key=lambda v: -adj[v][0][1])[0]
    steps, prev, cur = [], None, start
    while True:
        nxt = [(p, d, o) for p, d, o in adj[cur] if o != prev]
        if not nxt:
            break
        p, d, o = nxt[0]
        steps.append((cur, p, d, o))
        prev, cur = cur, o
    return steps if len(steps) == len(ve) else None


# ----------------------------------------------------------------------------- per building
def _run(store, q, box):
    try:
        box["rows"] = list(store.query(q, use_default_graph_as_union=True))
        box["vars"] = None
    except Exception as e:  # noqa: BLE001
        box["err"] = f"{type(e).__name__}: {e}"[:300]


def run_with_timeout(store, q, timeout=TIMEOUT):
    box = {}
    t = threading.Thread(target=_run, args=(store, q, box), daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    return box.get("rows"), box.get("err")


def building_meta(b):
    return json.loads((B2S / f"{b}.json").read_text(encoding="utf-8"))[0]


def abox_index(nt_path):
    types = collections.defaultdict(set)
    nodes = set()
    for t in ox.parse(path=str(nt_path), format=ox.RdfFormat.N_TRIPLES):
        if isinstance(t.object, ox.NamedNode):
            nodes.add(t.subject.value)
            if t.predicate.value == kg.RDF_TYPE:
                types[t.subject.value].add(t.object.value)
            else:
                nodes.add(t.object.value)
        else:
            nodes.add(t.subject.value)
    return types, nodes


def primary_type(types, anc):
    """Most specific asserted type (deepest in the hierarchy), ties broken lexically."""
    if not types:
        return None
    return sorted(types, key=lambda c: (-len(anc.get(c, {c})), c))[0]


def label_building(b, stratum, ttl, skip=()):
    nt = kg.nt_cache(b, ttl)
    tb = kg.tbox_path(stratum)
    store = kg.make_store(nt, tb)
    anc, _ = kg.subclass_closure(kg.tbox_triples(tb) + list(ox.parse(path=str(nt), format=ox.RdfFormat.N_TRIPLES)))
    types, nodes = abox_index(nt)
    meta = building_meta(b.split("@")[0])
    out, stats = [], collections.Counter()
    for q in meta["queries"]:
        rec = {"query_id": q["query_id"], "pattern_type": q["pattern_type"]}
        if q["query_id"] in skip:
            stats["timeout"] += 1
            continue
        try:
            edges, tpats, other, proj = parse_patterns(q["sparql_query"])
            star = select_star(q["sparql_query"])
        except Exception as e:  # noqa: BLE001
            stats["parse_error"] += 1
            continue
        yield ("progress", q["query_id"])
        rows, err = run_with_timeout(store, star)
        if err == "timeout":
            stats["timeout"] += 1
            continue
        if err or not rows:
            stats["error" if err else "empty"] += 1
            continue
        stats["ok"] += 1
        YV, YE, Yans = set(), set(), set()
        tclass = collections.defaultdict(set)
        for s, c in tpats:
            if isinstance(s, Variable):
                tclass[str(s)].add(str(c))
        path_counts = collections.Counter()
        chain = chain_order(edges)

        def val(sol, t):
            if isinstance(t, Variable):
                v = sol[str(t)]
                return None if v is None else v
            return ox.NamedNode(str(t)) if isinstance(t, URIRef) else None

        for sol in rows:
            for s, p, o in edges:
                sv, ov = val(sol, s), val(sol, o)
                if isinstance(sv, ox.NamedNode):
                    YV.add(sv.value)
                    if isinstance(ov, ox.NamedNode):
                        YE.add((sv.value, str(p), ov.value))
                        YV.add(ov.value)
            for s, c in tpats:
                sv = val(sol, s)
                if isinstance(sv, ox.NamedNode):
                    YV.add(sv.value)
                    YV.add(str(c))
                    for t in types.get(sv.value, ()):
                        if str(c) in anc.get(t, {t}):
                            YE.add((sv.value, kg.RDF_TYPE, t))
                            YV.add(t)
            for s, _, o in other:
                for t in (s, o):
                    v = val(sol, t)
                    if isinstance(v, ox.NamedNode):
                        YV.add(v.value)
            for v in proj:
                x = sol[v]
                if isinstance(x, ox.NamedNode):
                    Yans.add(x.value)
            if chain:
                key = []
                ok = True
                for i, (a, p, d, c) in enumerate(chain):
                    if i == 0:
                        av = sol[a]
                        key.append(primary_type(types.get(av.value, ()), anc) if isinstance(av, ox.NamedNode) else None)
                    cv = sol[c]
                    ct = primary_type(types.get(cv.value, ()), anc) if isinstance(cv, ox.NamedNode) else None
                    key += [p, d, ct]
                    ok &= ct is not None
                if ok and key[0] is not None:
                    path_counts[tuple(key)] += 1
        YV &= nodes | {c for cs in tclass.values() for c in cs} | {t for ts in types.values() for t in ts}
        p_ast = [[sorted(tclass.get(str(s), [None]))[0] if isinstance(s, Variable) else str(s), str(p),
                  sorted(tclass.get(str(o), [None]))[0] if isinstance(o, Variable) else str(o)]
                 for s, p, o in edges]
        sq = q["sparql_query"]
        weight = 0.5 if (stratum == "s223" and ("cnx" in sq or "hasConnectionPoint" in sq)) else 1.0
        rec.update({
            "weight": weight,
            "questions": [x["question_text"] for x in sorted(q["questions"], key=lambda x: x["question_number"])],
            "sources": [x["source"] for x in sorted(q["questions"], key=lambda x: x["question_number"])],
            "n_rows": len(rows), "truncated": len(rows) >= ROW_CAP,
            "Y_V": sorted(YV), "Y_E": sorted(YE), "Y_ans": sorted(Yans),
            "proj": proj, "P_ast": p_ast,
            "chain": [list(c) for c in chain] if chain else None,
            "P_inst": list(path_counts.most_common(1)[0][0]) if path_counts else None,
        })
        out.append(rec)
    rng = random.Random(0)
    n_exec = len(out)
    if len(out) > MAX_Q:
        out = sorted(rng.sample(out, MAX_Q), key=lambda r: r["query_id"])
    yield ("done", {"building": b, "stratum": stratum, "n_queries": len(meta["queries"]),
                    "n_exec": n_exec, "n_kept": len(out), "stats": dict(stats), "queries": out})


def _worker(b, stratum, ttl, skip, conn):
    try:
        for kind, payload in label_building(b, stratum, ttl, skip):
            conn.send((kind, payload))
    except Exception as e:  # noqa: BLE001
        conn.send(("fail", f"{type(e).__name__}: {e}"))
    conn.close()


def label_with_timeouts(b, stratum, ttl):
    """Run a building in a child process; if one query hangs past the timeout, kill and skip it."""
    skip = set()
    while True:
        parent, child = mp.Pipe(duplex=False)
        p = mp.Process(target=_worker, args=(b, stratum, ttl, tuple(skip), child), daemon=True)
        p.start()
        child.close()
        cur, last = None, time.time()
        while True:
            if parent.poll(5):
                try:
                    kind, payload = parent.recv()
                except EOFError:
                    p.join()
                    return {"building": b, "fail": "worker died", "stratum": stratum}
                if kind == "progress":
                    cur, last = payload, time.time()
                elif kind == "done":
                    p.join()
                    payload["stats"]["timeout"] = payload["stats"].get("timeout", 0)
                    return payload
                else:
                    p.join()
                    return {"building": b, "fail": payload, "stratum": stratum}
            elif cur and time.time() - last > TIMEOUT + 30:
                p.kill()
                skip.add(cur)
                print(f"  [{b}] hard-timeout on {cur}; restarting", flush=True)
                break
            elif not p.is_alive() and not parent.poll(0):
                return {"building": b, "fail": "worker exited", "stratum": stratum}


def _job(args):
    b, stratum, path = args
    t0 = time.time()
    res = label_with_timeouts(b, stratum, path)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main_build2sparql(buildings=None, procs=4):
    LDIR.mkdir(parents=True, exist_ok=True)
    man = kg.manifest()
    todo = []
    for b, r in sorted(man.items()):
        if buildings and b not in buildings:
            continue
        if (LDIR / f"{b}.json").exists() and not FORCE:
            continue
        path = ROOT / r["path"]
        comp = path.with_suffix(".compiled.ttl")
        todo.append((b, r["stratum"], str(path)))
        if comp.exists():
            todo.append((b + "@compiled", r["stratum"], str(comp)))
    results = collections.defaultdict(list)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(procs) as ex:
        for res in ex.map(_job, todo):
            name = res["building"]
            base = name.split("@")[0]
            results[base].append(res)
            print(f"{name}: {res.get('n_exec', 'FAIL')}/{res.get('n_queries', '?')} exec  {res.get('stats', res.get('fail'))}  {res['seconds']}s", flush=True)
    for base, rs in results.items():
        good = [r for r in rs if "fail" not in r]
        if not good:
            (LDIR / f"{base}.fail.json").write_text(json.dumps(rs, indent=1))
            continue
        best = max(good, key=lambda r: (r["n_exec"], "@" not in r["building"]))
        best["graph_variant"] = "compiled" if "@" in best["building"] else "raw"
        best["building"] = base
        (LDIR / f"{base}.json").write_text(json.dumps(best))


# ----------------------------------------------------------------------------- BuildingQA gold
STD_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "brick": "https://brickschema.org/schema/Brick#",
    "ref": "https://brickschema.org/schema/Brick/ref#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "s223": "http://data.ashrae.org/standard223#",
    "owl": "http://www.w3.org/2002/07/owl#",
}


def ttl_prefix_block(ttl):
    """Prefixes declared in a TTL file (as the BuildingQA ReAct baseline does), plus standard ones."""
    out = dict(STD_PREFIXES)
    with open(ttl, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*@?prefix\s+(\w*):\s*<([^>]+)>", line, re.I)
            if m:
                out[m.group(1)] = m.group(2)
            elif line.strip() and not line.lstrip().upper().startswith(("@", "#", "PREFIX")):
                break
    return out


def with_prefixes(sparql, prefixes):
    """Prepend PREFIX lines for prefixes used but not declared in the query."""
    declared = set(re.findall(r"PREFIX\s+(\w*):", sparql, re.I))
    body = re.sub(r"<[^>]*>|\"[^\"]*\"", " ", sparql)
    used = set(re.findall(r"(?<![\w?$])([A-Za-z][\w\-]*):[A-Za-z_]", body))
    add = [f"PREFIX {p}: <{prefixes[p]}>" for p in sorted(used - declared) if p in prefixes]
    return ("\n".join(add) + "\n" + sparql) if add else sparql


def main_buildingqa():
    bq = ROOT / "data" / "BuildingQA"
    out = ROOT / "runs" / "data" / "buildingqa_gold.json"
    gold = {}
    for f in sorted((bq / "Benchmark_QA_pairs").glob("*_combined.json")):
        recs = json.loads(f.read_text(encoding="utf-8"))
        for rec in recs:
            b = rec["building_id"].replace(".ttl", "")
            tb = kg.TBOX_223P if b == "b59" else kg.TBOX_BRICK
            store = kg.make_store(bq / "eval_buildings" / f"{b}.ttl", tb, abox_format=ox.RdfFormat.TURTLE)
            ttl_prefixes = ttl_prefix_block(bq / "eval_buildings" / f"{b}.ttl")
            for q in rec["queries"]:
                rows, err = run_with_timeout(store, with_prefixes(q["sparql_query"], ttl_prefixes), timeout=300)
                rows_out = []
                pv = re.findall(r"\?(\w+)", re.search(r"SELECT\s+(.+?)\s+WHERE", q["sparql_query"], re.I | re.S).group(1))
                for sol in rows or []:
                    d = {}
                    for v in dict.fromkeys(pv):
                        try:
                            x = sol[v]
                        except Exception:  # noqa: BLE001
                            x = None
                        if x is not None:
                            d[v] = {"value": kg.term_str(x)}
                    if d:
                        rows_out.append(d)
                gold[q["query_id"]] = {"building": b, "n_rows": len(rows_out), "error": err, "rows": rows_out}
                print(b, q["query_id"], len(rows_out), err or "", flush=True)
    out.write_text(json.dumps(gold))
    print("saved", out, len(gold))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--buildingqa", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--procs", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()
    FORCE = a.force
    if a.buildingqa:
        main_buildingqa()
    else:
        main_build2sparql(a.buildings, a.procs)
