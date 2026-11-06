"""Literal-aware label derivation (Y_V, Y_E, Y_ans, P_ast, chain, P_inst) shared by labels_v2 and synth.

Same rules as bridge.labels.label_building, plus:
  * a triple pattern (s, p, ?o) whose ?o binds a LITERAL maps to the attribute node
    lit:<sha1(s,p,lit)> (if kept by the 12-per-subject cap): the attr node goes to Y_V, the edge
    (s, p, attr) to Y_E, and if ?o is projected the attr node goes to Y_ans;
  * sequence property paths of plain IRIs (e.g. ?p ref:hasExternalReference/ref:hasTimeseriesId ?id)
    are resolved through the ABox so intermediate nodes/edges and the final literal/IRI are labelled;
  * chain / P_inst are computed over the ENTITY edges only (edges whose object variable binds a
    literal are excluded), so a query = entity chain + literal attributes still gets a path label;
  * Y_ans_cols {var: [node ids]} per projected variable (for per-column metrics).
"""
import collections

import pyoxigraph as ox
from rdflib import URIRef, Variable
from rdflib.paths import SequencePath

from bridge import kg
from bridge.labels import ROW_CAP, chain_order, parse_patterns, primary_type, run_with_timeout, select_star
from bridge.v2 import common as C


class Ctx:
    """Per-building context: store, types, ancestors, node set, out-adjacency, kept attr ids."""

    def __init__(self, b, nt=None, strat=None):
        self.b = b
        self.stratum = strat or C.stratum(b)
        self.nt = nt or C.nt_path(b)
        tb = kg.tbox_path(self.stratum)
        self.abox = C.read_abox(self.nt)
        self.store = kg.make_store(self.nt, tb)
        self.anc, _ = kg.subclass_closure(kg.tbox_triples(tb) + self.abox)
        self.types = collections.defaultdict(set)
        self.nodes = set()
        self.out = collections.defaultdict(list)       # (s, p) -> [object terms]
        for t in self.abox:
            s, p, o = t.subject.value, t.predicate.value, t.object
            self.nodes.add(s)
            if isinstance(o, ox.NamedNode):
                if p == kg.RDF_TYPE:
                    self.types[s].add(o.value)
                else:
                    self.nodes.add(o.value)
            self.out[(s, p)].append(o)
        self.attr_ok = C.kept_attr_ids(self.abox)
        self.prim = {s: primary_type(ts, self.anc) for s, ts in self.types.items()}


def _seq_iris(p):
    if isinstance(p, SequencePath) and all(isinstance(a, URIRef) for a in p.args) and len(p.args) <= 3:
        return [str(a) for a in p.args]
    return None


def label_query(ctx, sparql, timeout=60, rows=None):
    """Returns (record dict, None) or (None, reason)."""
    try:
        edges, tpats, other, proj = parse_patterns(sparql)
        star = select_star(sparql)
    except Exception as e:  # noqa: BLE001
        return None, f"parse_error: {type(e).__name__}"
    if rows is None:
        rows, err = run_with_timeout(ctx.store, star, timeout)
        if err == "timeout":
            return None, "timeout"
        if err:
            return None, "error"
    if not rows:
        return None, "empty"
    types, anc = ctx.types, ctx.anc
    YV, YE, Yans = set(), set(), set()
    cols = collections.defaultdict(set)
    tclass = collections.defaultdict(set)
    for s, c in tpats:
        if isinstance(s, Variable):
            tclass[str(s)].add(str(c))

    def val(sol, t):
        if isinstance(t, Variable):
            try:
                return sol[str(t)]
            except (KeyError, ValueError):
                return None
        return ox.NamedNode(str(t)) if isinstance(t, URIRef) else None

    # variables that bind literals (in any row) -> excluded from the entity chain
    lit_vars = set()
    for sol in rows[:200]:
        for s, p, o in edges:
            if isinstance(o, Variable) and isinstance(val(sol, o), ox.Literal):
                lit_vars.add(str(o))
    ent_edges = [e for e in edges if not (isinstance(e[2], Variable) and str(e[2]) in lit_vars)]
    chain = chain_order(ent_edges)
    # fallback chain over edges between query-typed variables (drops untyped attribute leaves)
    typed_edges = [e for e in ent_edges if isinstance(e[0], Variable) and isinstance(e[2], Variable)
                   and str(e[0]) in tclass and str(e[2]) in tclass]
    chain2 = chain_order(typed_edges) if len(typed_edges) < len(ent_edges) else None
    path_counts = collections.Counter()
    path_counts2 = collections.Counter()
    seq_other = [(s, _seq_iris(p), o) for s, p, o in other]
    for sol in rows:
        litmap = {}
        for s, p, o in edges:
            sv, ov = val(sol, s), val(sol, o)
            if isinstance(sv, ox.NamedNode):
                YV.add(sv.value)
                if isinstance(ov, ox.NamedNode):
                    YE.add((sv.value, str(p), ov.value))
                    YV.add(ov.value)
                elif isinstance(ov, ox.Literal):
                    a = C.attr_id(sv.value, str(p), ov.value)
                    if a in ctx.attr_ok:
                        YV.add(a)
                        YE.add((sv.value, str(p), a))
                        if isinstance(o, Variable):
                            litmap[str(o)] = a
        for s, c in tpats:
            sv = val(sol, s)
            if isinstance(sv, ox.NamedNode):
                YV.add(sv.value)
                YV.add(str(c))
                for t in types.get(sv.value, ()):
                    if str(c) in anc.get(t, {t}):
                        YE.add((sv.value, kg.RDF_TYPE, t))
                        YV.add(t)
        for s, seq, o in seq_other:
            sv, ov = val(sol, s), val(sol, o)
            for v in (sv, ov):
                if isinstance(v, ox.NamedNode):
                    YV.add(v.value)
            if seq and isinstance(sv, ox.NamedNode) and ov is not None:
                # walk s -p1-> m1 (-p2-> m2) -pk-> o through the ABox
                frontier = [(sv.value, [])]
                for p_ in seq[:-1]:
                    nxt = []
                    for n, tr in frontier:
                        for m in ctx.out.get((n, p_), ()):
                            if isinstance(m, ox.NamedNode):
                                nxt.append((m.value, tr + [(n, p_, m.value)]))
                    frontier = nxt[:50]
                for n, tr in frontier:
                    hit = None
                    for m in ctx.out.get((n, seq[-1]), ()):
                        if m == ov:
                            if isinstance(m, ox.Literal):
                                a = C.attr_id(n, seq[-1], m.value)
                                hit = a if a in ctx.attr_ok else None
                            else:
                                hit = m.value
                            break
                    if hit:
                        for (x, p_, y) in tr + [(n, seq[-1], hit)]:
                            YE.add((x, p_, y)); YV.add(x); YV.add(y)
                        if isinstance(o, Variable) and isinstance(ov, ox.Literal):
                            litmap[str(o)] = hit
                        break
        for v in proj:
            try:
                x = sol[v]
            except (KeyError, ValueError):
                x = None
            if isinstance(x, ox.NamedNode):
                Yans.add(x.value); cols[v].add(x.value)
            elif isinstance(x, ox.Literal) and v in litmap:
                Yans.add(litmap[v]); cols[v].add(litmap[v])
        for ch, pc in ((chain, path_counts), (chain2, path_counts2)):
            if not ch:
                continue
            key, ok = [], True
            for i, (a, p, d, c) in enumerate(ch):
                if i == 0:
                    av = val(sol, Variable(a))
                    key.append(primary_type(types.get(av.value, ()), anc) if isinstance(av, ox.NamedNode) else None)
                cv = val(sol, Variable(c))
                ct = primary_type(types.get(cv.value, ()), anc) if isinstance(cv, ox.NamedNode) else None
                key += [p, d, ct]
                ok &= ct is not None
            if ok and key[0] is not None:
                pc[tuple(key)] += 1
    if not path_counts and path_counts2:
        chain, path_counts = chain2, path_counts2
    known_cls = {c for cs in tclass.values() for c in cs} | {t for ts in types.values() for t in ts}
    YV = {n for n in YV if n in ctx.nodes or n in known_cls or C.is_attr(n)}
    p_ast = [[sorted(tclass.get(str(s), [None]))[0] if isinstance(s, Variable) else str(s), str(p),
              sorted(tclass.get(str(o), [None]))[0] if isinstance(o, Variable) else str(o)]
             for s, p, o in edges]
    return {
        "n_rows": len(rows), "truncated": len(rows) >= ROW_CAP,
        "Y_V": sorted(YV), "Y_E": sorted(YE), "Y_ans": sorted(Yans),
        "Y_ans_cols": {v: sorted(cols[v]) for v in proj},
        "lit_vars": sorted(v for v in proj if v in lit_vars),
        "proj": proj, "P_ast": p_ast,
        "chain": [list(c) for c in chain] if chain else None,
        "P_inst": list(path_counts.most_common(1)[0][0]) if path_counts else None,
    }, None


def s223_weight(stratum, sparql):
    """v1 down-weighting of 223P connection-topology queries."""
    return 0.5 if (stratum == "s223" and ("cnx" in sparql or "hasConnectionPoint" in sparql)) else 1.0
