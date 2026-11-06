"""
Graph utilities for BKGQA (ontology-agnostic port of BrickQA/brick_utils.py).

- KGGraph: owns one rdflib Graph + its OntologyProfile (no process-wide singleton).
- execute_sparql(): same threaded-timeout behaviour and status enum as BrickQA,
  plus auto-insertion of PREFIX declarations that are used but not declared.
- Tools: search_graph(), get_entity(), get_property_examples() + formatters.
  All URIs in tool outputs are compacted with the profile's prefix map.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from .profile import (
    LABEL_PREDICATES, META_CLASSES, OntologyProfile, build_profile, format_literal,
    local_name, split_words, REF_NS,
)


class SparqlExecutionStatus(Enum):
    """Status codes for SPARQL execution (same values as BrickQA)."""
    SUCCESS = "success"
    EMPTY_RESULT = "empty_result"
    SYNTAX_ERROR = "syntax_error"
    TIMED_OUT = "timed_out"
    OTHER_ERROR = "other_error"

    def get_message(self) -> str:
        messages = {
            self.SUCCESS: "Query executed successfully",
            self.EMPTY_RESULT: "Query returned no results",
            self.SYNTAX_ERROR: "Query has syntax error",
            self.TIMED_OUT: "Query execution timed out",
            self.OTHER_ERROR: "Query execution encountered an error",
        }
        return messages.get(self, "Unknown status")


# --------------------------------------------------------------------------
# PREFIX handling
# --------------------------------------------------------------------------
_IRI_RE = re.compile(r"<[^<>\s\"{}|^`\\]*>")
_STR_RE = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')
_DECL_RE = re.compile(r"PREFIX\s+([A-Za-z][\w\-.]*)?:\s*<", re.IGNORECASE)
_PNAME_RE = re.compile(r"(?<![\w?$:\-.])([A-Za-z](?:[\w\-.]*[\w\-])?):(?=[\w\-%]|\s|$|[;,.)\]}])")


def strip_iris_and_strings(query: str) -> str:
    q = _STR_RE.sub('""', query)
    q = _IRI_RE.sub("<>", q)
    q = re.sub(r"#[^\n]*", "", q)  # comments (IRIs already removed)
    return q


def declared_prefixes(query: str) -> Dict[str, str]:
    return {m.group(1) or "": m.group(2) for m in
            re.finditer(r"PREFIX\s+([A-Za-z][\w\-.]*)?:\s*<([^>]*)>", query, re.IGNORECASE)}


def used_prefixes(query: str) -> List[str]:
    body = strip_iris_and_strings(query)
    body = _DECL_RE.sub(" ", body)
    seen = []
    for m in _PNAME_RE.finditer(body):
        p = m.group(1)
        if p not in seen:
            seen.append(p)
    return seen


def add_missing_prefixes(query: str, profile: Optional[OntologyProfile]) -> Tuple[str, List[str]]:
    """Prepend PREFIX lines for prefixes used in the query but not declared (known ones only)."""
    if profile is None:
        return query, []
    declared = declared_prefixes(query)
    missing = [p for p in used_prefixes(query) if p not in declared and p in profile.prefixes]
    if not missing:
        return query, []
    header = "\n".join(f"PREFIX {p}: <{profile.prefixes[p]}>" for p in missing)
    return header + "\n" + query.lstrip(), missing


# --------------------------------------------------------------------------
# KGGraph
# --------------------------------------------------------------------------
class KGGraph:
    """One building knowledge graph + its ontology profile. Not a singleton."""

    def __init__(self, ttl_file: Optional[str] = None, graph: Optional[Graph] = None,
                 store: str = "default", build_profile_now: bool = True):
        self.store = store
        self.ttl_file = ttl_file
        self.graph: Graph = graph if graph is not None else self._new_graph()
        self.profile: Optional[OntologyProfile] = None
        self.last_error: Optional[str] = None
        self._index = None
        self._search_cache: Dict[Tuple, List[Dict]] = {}
        self._entity_cache: Dict[str, Dict] = {}
        if ttl_file:
            self.load(ttl_file)
        if build_profile_now and (ttl_file or graph is not None):
            self.refresh_profile()

    def _new_graph(self) -> Graph:
        if self.store and self.store.lower() == "oxigraph":
            return Graph(store="Oxigraph")
        return Graph()

    def load(self, ttl_file: str, fmt: str = "turtle"):
        self.ttl_file = ttl_file
        self.graph.parse(ttl_file, format=fmt)
        print(f"Loaded {len(self.graph)} triples from {ttl_file}")
        self._invalidate()

    def refresh_profile(self):
        self.profile = build_profile(self.graph, self.ttl_file)
        self._invalidate(keep_profile=True)
        return self.profile

    def _invalidate(self, keep_profile: bool = False):
        self._index = None
        self._search_cache.clear()
        self._entity_cache.clear()
        if not keep_profile:
            self.profile = None

    def __len__(self):
        return len(self.graph)

    # ---------------- timeseries (optional, CSV -> observation triples) ----------------
    def add_timeseries_data(self, csv_file: str, max_rows: int = 100, start_date: str = None,
                            end_date: str = None, datetime_col: str = "Datetime",
                            datetime_format: str = "%m/%d/%Y %H:%M",
                            entity_namespace: Optional[str] = None,
                            obs_namespace: str = REF_NS):
        """Load wide-format CSV timeseries as ?entity ref:hasObservation ?obs triples.

        Column names are mapped to entity IRIs in `entity_namespace` (defaults to the most
        common namespace of typed instances in the graph).
        """
        import pandas as pd
        from datetime import datetime

        if self.profile is None:
            self.refresh_profile()
        ns = entity_namespace or self.profile.instance_namespace
        if not ns:
            raise ValueError("Cannot infer entity namespace for timeseries columns; pass entity_namespace")
        df_full = pd.read_csv(csv_file)
        df_full["_parsed_dt"] = pd.to_datetime(df_full[datetime_col], format=datetime_format)
        if start_date:
            df_full = df_full[df_full["_parsed_dt"] >= datetime.strptime(start_date, "%Y-%m-%d")]
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            df_full = df_full[df_full["_parsed_dt"] <= end_dt]
        df = df_full.tail(max_rows)
        has_obs, has_ts, has_val = (URIRef(obs_namespace + x) for x in ("hasObservation", "hasTimestamp", "hasValue"))
        for idx, row in df.iterrows():
            ts = row["_parsed_dt"].strftime("%Y-%m-%dT%H:%M:%S")
            for col in df.columns:
                if col in (datetime_col, "_parsed_dt"):
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                obs = URIRef(f"{ns}obs_{col}_{idx}")
                self.graph.add((URIRef(ns + str(col)), has_obs, obs))
                self.graph.add((obs, has_ts, Literal(ts, datatype=XSD.dateTime)))
                try:
                    self.graph.add((obs, has_val, Literal(float(val), datatype=XSD.float)))
                except (TypeError, ValueError):
                    self.graph.add((obs, has_val, Literal(str(val))))
        print(f"Added timeseries data from {csv_file}. Total triples: {len(self.graph)}")
        self.refresh_profile()

    def get_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        if self.profile is None or not self.profile.has_observations:
            return (None, None)
        ts = self.profile.obs_predicates["timestamp"]
        q = f"SELECT (MIN(?t) AS ?a) (MAX(?t) AS ?b) WHERE {{ ?o <{ts}> ?t }}"
        try:
            rows = list(self.graph.query(q))
            if rows and rows[0][0] is not None:
                return (str(rows[0][0])[:10], str(rows[0][1])[:10])
        except Exception as e:
            print(f"Warning: Could not determine date range: {e}")
        return (None, None)

    # ---------------- delegates ----------------
    def execute_sparql(self, query: str, **kw):
        return execute_sparql(self, query, **kw)

    def search(self, term: str, limit: int = 10):
        return search_graph(self, term, limit=limit)

    def get_entity(self, term: str):
        return get_entity(self, term)

    def get_property_examples(self, prop: str, limit: int = 5):
        return get_property_examples(self, prop, limit=limit)

    # ---------------- helpers used by tools ----------------
    def compact(self, node) -> str:
        if isinstance(node, Literal):
            return format_literal(node)
        if isinstance(node, BNode):
            return "[blank node]"
        return self.profile.compact(node) if self.profile else f"<{node}>"

    def types_of(self, node) -> List[str]:
        return [str(t) for t in self.graph.objects(node, RDF.type)]

    def primary_type(self, node) -> Optional[str]:
        ts = self.types_of(node)
        if not ts:
            return None
        cc = self.profile.class_counts if self.profile else Counter()
        return min(ts, key=lambda t: (cc.get(t, 10**9), -len(t)))

    def node_exists(self, iri: str) -> bool:
        u = URIRef(iri)
        return ((u, None, None) in self.graph or (None, None, u) in self.graph
                or (None, u, None) in self.graph)

    # ---------------- search index ----------------
    def index(self):
        if self._index is not None:
            return self._index
        g = self.graph
        classes, preds, subjects = set(), Counter(), set()
        schema = set()
        labels: Dict[object, List[str]] = defaultdict(list)
        bnode_names: Dict[BNode, List[Tuple[str, str]]] = defaultdict(list)
        bnode_parent: Dict[BNode, Tuple[object, str]] = {}
        prop_types = {str(RDF.Property), "http://www.w3.org/2002/07/owl#ObjectProperty",
                      "http://www.w3.org/2002/07/owl#DatatypeProperty"}
        for s, p, o in g:
            preds[p] += 1
            if p == RDF.type:
                classes.add(o)
                if str(o) in META_CLASSES:
                    schema.add(s)
                if str(o) in prop_types:
                    preds[s] += 0
            elif p == RDFS.subClassOf:
                classes.add(s)
                schema.add(s)
            if isinstance(o, Literal):
                ps = str(p)
                ln = local_name(ps).lower()
                if ps in LABEL_PREDICATES or "label" in ln or "name" in ln:
                    if isinstance(s, BNode):
                        bnode_names[s].append((ps, str(o)))
                    elif len(labels[s]) < 3:
                        labels[s].append(str(o))
            if isinstance(o, BNode) and o not in bnode_parent and isinstance(s, URIRef):
                bnode_parent[o] = (s, str(p))
            if isinstance(s, URIRef):
                subjects.add(s)

        entries = []  # (node, category, local, words, labels)

        def add(node, cat):
            loc = local_name(str(node))
            labs = labels.get(node, [])
            words = " ".join(split_words(loc) + [w for l in labs for w in split_words(l)])
            entries.append((node, cat, loc, words, labs))

        for c in classes:
            if isinstance(c, URIRef):
                add(c, "class")
        for p in preds:
            if isinstance(p, URIRef) and p not in classes:
                add(p, "predicate")
        for s in subjects:
            if s not in classes and s not in preds and s not in schema:
                add(s, "instance")
        # blank nodes carrying name-like literals (e.g. external references) -> report the parent
        for b, names in bnode_names.items():
            parent = bnode_parent.get(b)
            if parent is None:
                continue
            for ps, val in names[:3]:
                entries.append((parent[0], "instance", local_name(str(parent[0])),
                                " ".join(split_words(val)), [val], (parent[1], ps, val)))
        local_index: Dict[str, List] = defaultdict(list)
        for e in entries:
            if len(e) == 5:
                local_index[e[2].lower()].append(e[0])
        self._index = {"entries": entries, "local": local_index, "pred_counts": preds, "labels": labels}
        return self._index


# --------------------------------------------------------------------------
# SPARQL execution
# --------------------------------------------------------------------------
def execute_sparql(kg: KGGraph, query: str, return_status: bool = False, timeout: int = 180,
                   result_limit: int = None):
    """Execute a SPARQL query on the graph with timeout enforcement (BrickQA semantics).

    Returns list of {var: str(value)} (or (results, status) if return_status).
    The last error message (if any) is stored in kg.last_error.
    """
    kg.last_error = None
    try:
        graph = kg.graph
        if result_limit and not re.search(r"LIMIT\s+\d+", query, re.IGNORECASE):
            query = query.rstrip().rstrip(";") + f" LIMIT {result_limit}"
            print(f"[SPARQL] Auto-added LIMIT {result_limit} to query")

        query, added = add_missing_prefixes(query, kg.profile)
        if added:
            print(f"[SPARQL] Auto-added missing PREFIX declarations: {', '.join(added)}")

        preview = query[:150].replace("\n", " ") + ("..." if len(query) > 150 else "")
        print(f"[SPARQL] Executing query: {preview}")
        print(f"[SPARQL] Graph size: {len(graph)} triples")
        print(f"[SPARQL] Timeout: {timeout} seconds")
        start = time.time()

        query_result = [None]
        query_exception = [None]

        def run_query():
            try:
                res = graph.query(query)
                # materialise inside the thread so lazy evaluation is covered by the timeout
                query_result[0] = (list(res.vars or []), list(res))
            except Exception as e:  # noqa: BLE001
                query_exception[0] = e

        th = threading.Thread(target=run_query, daemon=True)
        th.start()
        th.join(timeout=timeout)
        elapsed = time.time() - start

        if th.is_alive():
            print(f"[SPARQL] Query TIMEOUT after {timeout} seconds!")
            kg.last_error = f"Query timed out after {timeout} seconds. Simplify the query or add LIMIT."
            status = SparqlExecutionStatus.TIMED_OUT
            return ([], status) if return_status else []

        if query_exception[0]:
            raise query_exception[0]

        vars_, rows = query_result[0]
        print(f"[SPARQL] Query completed in {elapsed:.2f} seconds")
        if vars_:
            result_list = [{str(v): str(row[v]) for v in vars_ if row[v] is not None} for row in rows]
        else:  # ASK / CONSTRUCT
            result_list = [{"result": str(r)} for r in rows]
        result_list = [r for r in result_list if r]
        status = SparqlExecutionStatus.SUCCESS if result_list else SparqlExecutionStatus.EMPTY_RESULT
        return (result_list, status) if return_status else result_list

    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "syntax" in msg or "parse" in msg or "expected" in msg:
            status = SparqlExecutionStatus.SYNTAX_ERROR
        elif "timeout" in msg:
            status = SparqlExecutionStatus.TIMED_OUT
        else:
            status = SparqlExecutionStatus.OTHER_ERROR
        kg.last_error = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"[SPARQL] Execution error: {e}")
        return ([], status) if return_status else []


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def _clean_arg(term: str) -> str:
    t = (term or "").strip()
    for _ in range(2):
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'`":
            t = t[1:-1].strip()
    if t.lower().startswith(("search_term=", "entity_id=", "uri=", "property=", "term=")):
        t = _clean_arg(t.split("=", 1)[1])
    return t


def _literal_attrs(kg: KGGraph, node, exclude=(), n: int = 3) -> List[Tuple[str, str]]:
    out = []
    for p, o in kg.graph.predicate_objects(node):
        if isinstance(o, Literal) and str(p) not in exclude:
            out.append((kg.compact(p), format_literal(o, 50)))
            if len(out) >= n:
                break
    return out


def search_graph(kg: KGGraph, search_term: str, limit: int = 10) -> List[Dict]:
    """Search classes, predicates (incl. literal predicates) and instances by local name / label."""
    term = _clean_arg(search_term)
    key = (term.lower(), limit)
    if key in kg._search_cache:
        return kg._search_cache[key]
    if not term:
        return []
    idx = kg.index()
    prof = kg.profile
    tl = term.lower()
    t_words = split_words(term)
    t_phrase = " ".join(t_words)
    expanded = prof.expand(term) if prof else None

    scored = []
    for e in idx["entries"]:
        node, cat, loc, words, labs = e[:5]
        via = e[5] if len(e) > 5 else None
        ll = loc.lower()
        score = 0
        if (expanded and str(node) == expanded and via is None) or ll == tl or any(l.lower() == tl for l in labs):
            score = 4
        elif t_phrase and (f" {t_phrase} " in f" {words} "):
            score = 3
        elif tl in ll or any(tl in l.lower() for l in labs):
            score = 2
        elif t_words and all(w in words for w in t_words):
            score = 1
        if score:
            scored.append((score, cat, node, labs, via))

    cc = prof.class_counts if prof else Counter()
    pc = idx["pred_counts"]

    def usage(cat, node):
        return cc.get(str(node), 0) if cat == "class" else pc.get(node, 0) if cat == "predicate" else 0

    caps = {"class": 6, "predicate": 6, "instance": limit}
    results: List[Dict] = []
    for cat in ("class", "predicate", "instance"):
        items = [x for x in scored if x[1] == cat]
        items.sort(key=lambda x: (-x[0], -(usage(cat, x[2]) > 0), -usage(cat, x[2]), len(str(x[2]))))
        seen = set()
        for score, _, node, labs, via in items:
            if node in seen:
                continue
            seen.add(node)
            r = {"id": kg.compact(node), "uri": str(node), "category": cat, "label": labs[0] if labs else "",
                 "score": score}
            if cat == "class":
                r["count"] = cc.get(str(node), 0)
            elif cat == "predicate":
                r["count"] = pc.get(node, 0)
                r["literal"] = str(node) in (prof.attr_pred_counts if prof else {})
            else:
                r["types"] = [kg.compact(t) for t in kg.types_of(node)][:4]
                r["attributes"] = _literal_attrs(kg, node, exclude={str(RDFS.label)})
                if via:
                    r["via"] = f"{kg.compact(URIRef(via[0]))}/{kg.compact(URIRef(via[1]))} = {format_literal(Literal(via[2]), 50)}"
            results.append(r)
            if len(seen) >= caps[cat]:
                break
    kg._search_cache[key] = results
    return results


def format_search_results(results: List[Dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for cat, title in (("class", "Classes"), ("predicate", "Predicates"), ("instance", "Instances")):
        items = [r for r in results if r["category"] == cat]
        if not items:
            continue
        lines.append(f"{title}:")
        for r in items:
            lab = f' label "{r["label"]}"' if r.get("label") else ""
            if cat == "class":
                lines.append(f"  {r['id']}  [class; {r['count']} instances]{lab}")
            elif cat == "predicate":
                kind = "literal attribute" if r.get("literal") else "relationship"
                lines.append(f"  {r['id']}  [{kind}; {r['count']} uses]{lab}")
            else:
                types = ", ".join(r.get("types") or []) or "untyped"
                lines.append(f"  {r['id']}  type: {types}{lab}")
                if r.get("via"):
                    lines.append(f"     matched via {r['via']}")
                if r.get("attributes"):
                    lines.append("     attrs: " + "; ".join(f"{p}={v}" for p, v in r["attributes"]))
    return "\n".join(lines)


def resolve_term(kg: KGGraph, term: str, kind: str = "node") -> Tuple[Optional[str], str]:
    """Resolve 'prefix:local', '<iri>', full IRI or bare local name -> (iri, note)."""
    t = _clean_arg(term)
    if not t:
        return None, "empty argument"
    prof = kg.profile
    iri = prof.expand(t) if prof else (t[1:-1] if t.startswith("<") else None)
    if iri and kg.node_exists(iri):
        return iri, ""
    idx = kg.index()
    cands = list(idx["local"].get(t.lower(), []))
    if not cands and ":" in t:
        cands = list(idx["local"].get(t.split(":", 1)[1].lower(), []))
    if kind == "predicate":
        cands = [c for c in cands if c in idx["pred_counts"]] or \
                [p for p in idx["pred_counts"] if local_name(str(p)).lower() == local_name(t).lower()]
    if cands:
        cands = sorted(set(cands), key=lambda c: -(idx["pred_counts"].get(c, 0) if kind == "predicate"
                                                    else len(list(kg.graph.predicate_objects(c)))))
        note = ""
        if len(cands) > 1:
            note = "Other matches: " + ", ".join(kg.compact(c) for c in cands[1:6])
        if iri and str(cands[0]) != iri:
            note = (f"{t} not found; using {kg.compact(cands[0])}. " + note).strip()
        return str(cands[0]), note
    return iri, "not found in graph"


def get_entity(kg: KGGraph, entity_id: str, max_per_pred: int = 5) -> Dict:
    """Outgoing and incoming triples (incl. literals) of an entity, compacted."""
    key = _clean_arg(entity_id)
    if key in kg._entity_cache:
        return kg._entity_cache[key]
    iri, note = resolve_term(kg, key)
    info = {"entity": key, "uri": iri, "found": False, "types": [], "outgoing": [], "incoming": [],
            "instances": None, "note": note}
    if not iri or not kg.node_exists(iri):
        hits = search_graph(kg, key.split(":")[-1], limit=5) if key else []
        if hits:
            info["note"] = (note + ". Similar: " + ", ".join(h["id"] for h in hits[:6])).strip(". ")
        kg._entity_cache[key] = info
        return info
    node = URIRef(iri)
    info["found"] = True
    info["entity"] = kg.compact(node)
    info["types"] = [kg.compact(t) for t in kg.types_of(node)]
    g = kg.graph

    grouped = defaultdict(list)
    for p, o in g.predicate_objects(node):
        if p == RDF.type:
            continue
        grouped[p].append(o)
    for p, objs in grouped.items():
        vals = []
        for o in objs[:max_per_pred]:
            if isinstance(o, Literal):
                vals.append(("literal", format_literal(o, 80)))
            elif isinstance(o, BNode):
                inner = []
                for bt in g.objects(o, RDF.type):
                    inner.append(f"a {kg.compact(bt)}")
                for bp, bo in g.predicate_objects(o):
                    if bp != RDF.type:
                        inner.append(f"{kg.compact(bp)} {kg.compact(bo)}")
                vals.append(("bnode", "[ " + " ; ".join(inner[:6]) + " ]"))
            else:
                pt = kg.primary_type(o)
                vals.append(("uri", kg.compact(o) + (f" ({kg.compact(URIRef(pt))})" if pt else "")))
        info["outgoing"].append((kg.compact(p), vals, max(0, len(objs) - max_per_pred)))

    inc = defaultdict(list)
    n_type = 0
    type_examples = []
    for s, p in g.subject_predicates(node):
        if p == RDF.type:
            n_type += 1
            if len(type_examples) < 5:
                type_examples.append(kg.compact(s))
            continue
        inc[p].append(s)
    for p, subs in inc.items():
        vals = []
        for s in subs[:max_per_pred]:
            pt = kg.primary_type(s) if not isinstance(s, BNode) else None
            vals.append(kg.compact(s) + (f" ({kg.compact(URIRef(pt))})" if pt else ""))
        info["incoming"].append((kg.compact(p), vals, max(0, len(subs) - max_per_pred)))
    if n_type:
        info["instances"] = (n_type, type_examples)
    kg._entity_cache[key] = info
    return info


def format_entity_info(info: Dict, max_lines: int = 70) -> str:
    if not info.get("found"):
        return f"Entity {info.get('entity')} not found in the graph. {info.get('note', '')}".strip()
    lines = [f"Entity: {info['entity']}"]
    if info.get("note"):
        lines.append(f"Note: {info['note']}")
    if info.get("types"):
        lines.append(f"Types: {', '.join(info['types'])}")
    if info["outgoing"]:
        lines.append("Outgoing:")
        for p, vals, more in info["outgoing"]:
            for kind, v in vals:
                lines.append(f"  {p} {'=' if kind == 'literal' else '->'} {v}")
            if more:
                lines.append(f"  {p} ... (+{more} more)")
    if info.get("instances"):
        n, ex = info["instances"]
        lines.append(f"Instances (rdf:type this, {n}): {', '.join(ex)}")
    if info["incoming"]:
        lines.append("Incoming:")
        for p, vals, more in info["incoming"]:
            for v in vals:
                lines.append(f"  {v} {p} -> this")
            if more:
                lines.append(f"  ... (+{more} more via {p})")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines omitted)"]
    return "\n".join(lines)


def get_property_examples(kg: KGGraph, property_name: str, limit: int = 5) -> Dict:
    """Examples of how a predicate is used, compacted, with subject/object types."""
    iri, note = resolve_term(kg, property_name, kind="predicate")
    out = {"property": _clean_arg(property_name), "uri": iri, "count": 0, "examples": [], "note": note}
    if not iri:
        return out
    p = URIRef(iri)
    out["property"] = kg.compact(p)
    n = 0
    for s, o in kg.graph.subject_objects(p):
        n += 1
        if len(out["examples"]) < limit:
            st = kg.primary_type(s) if not isinstance(s, BNode) else None
            sc = kg.compact(s) + (f" ({kg.compact(URIRef(st))})" if st else "")
            if isinstance(o, Literal):
                oc = format_literal(o, 60)
            else:
                ot = kg.primary_type(o) if not isinstance(o, BNode) else None
                oc = kg.compact(o) + (f" ({kg.compact(URIRef(ot))})" if ot else "")
            out["examples"].append((sc, out["property"], oc))
    out["count"] = n
    return out


def format_property_examples(ex: Dict) -> str:
    if not ex.get("examples"):
        return f"No examples found for {ex.get('property')}. {ex.get('note', '')}".strip()
    lines = [f"{ex['property']} is used {ex['count']} times. Examples:"]
    if ex.get("note"):
        lines.insert(0, f"Note: {ex['note']}")
    lines += [f"{s} -- {p} --> {o}" for s, p, o in ex["examples"]]
    return "\n".join(lines)


def format_results_table(rows: List[Dict], profile: Optional[OntologyProfile] = None, max_rows: int = 10) -> str:
    """Text table of SPARQL results (IRIs compacted for display only)."""
    import pandas as pd
    if not rows:
        return "No results"
    disp = rows
    if profile is not None:
        def cv(v):
            return profile.compact(v) if v.startswith(("http://", "https://", "urn:")) else v
        disp = [{k: cv(v) for k, v in r.items()} for r in rows]
    df = pd.DataFrame(disp)
    if len(df) > max_rows:
        half = max_rows // 2
        return (df.head(half).to_string(index=False) + f"\n... ({len(df) - max_rows} rows omitted) ...\n"
                + df.tail(max_rows - half).to_string(index=False, header=False) + f"\n(total rows: {len(df)})")
    return df.to_string(index=False)
