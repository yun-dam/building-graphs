"""Ontology-term filter (cf. BSchema remove-ontology.py): drop from an ABox every triple whose SUBJECT
is a term of a known ontology / vocabulary, so that embedded TBoxes (e.g. the Brick ontology inside
BuildingQA bldg11.ttl, 223P / QUDT / SHACL terms inside b59.ttl) do not become "instances".

A subject is removed if
  (1) its IRI is in a vocabulary namespace (VOCAB_PREFIXES; 223P: any http://data.ashrae.org/standard223*
      IRI EXCEPT the instance-data space http://data.ashrae.org/standard223/data/...), or
  (2) it is typed with a schema type (owl:Class, owl:*Property, owl:Ontology, owl:Restriction, rdf:Property,
      rdfs:Class, rdfs:Datatype, sh:NodeShape, sh:PropertyShape, sh:*Rule, ...), or
  (3) it is a (skolemized) blank node referenced ONLY by removed subjects (shape / restriction / list
      structure of removed terms), iterated to a fixpoint.
Objects are never touched: values such as unit:DEG_F, s223:Fluid-Air or class IRIs are KEPT.

python -m bridge.hybrid.clean [--bq] [--b2s] [--force]
Output: runs/hybrid/nt/{building}.nt (Build2SPARQL, graph variant as chosen by v1),
        runs/hybrid/nt/bq_{building}.nt (BuildingQA eval graphs), runs/hybrid/clean_report.json
"""
import argparse
import collections
import json
import re
import time

import pyoxigraph as ox

from bridge import kg
from bridge.hybrid import common as H
from bridge.v2 import common as C2

VOCAB_PREFIXES = (
    "http://data.ashrae.org/bacnet/",
    "http://www.w3.org/ns/shacl#", kg.OWL, "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#", "http://www.w3.org/2001/XMLSchema#",
    "http://www.w3.org/2004/02/skos/core#", "http://www.w3.org/ns/sosa/", "http://www.w3.org/ns/ssn/",
    "http://www.w3.org/ns/prov#", "http://www.w3.org/ns/dcat#", "http://purl.org/dc/terms/",
    "http://purl.org/dc/elements/1.1/", "http://purl.org/vocab/vann/", "http://xmlns.com/foaf/0.1/",
    "http://qudt.org/", "https://qudt.org/", "http://www.linkedmodel.org/", "http://voag.linkedmodel.org/",
    "https://w3id.org/rec", "https://w3id.org/bot#", "https://project-haystack.org/", "http://project-haystack.org/",
)
S223_BASE = "http://data.ashrae.org/standard223"
# Brick vocabularies: Brick#, Brick/ref#, BrickShape#, BrickTag#, and versioned .../schema/1.4/Brick#; NOT
# example data that happens to live under brickschema.org (e.g. .../schema/1.0.2/building_example#)
BRICK_RE = re.compile(r"^https?://brickschema\.org/schema/(\d[\d.]*/)?(Brick|BrickShape|BrickTag)(/ref|/tag)?([#/]|$)")
SH = "http://www.w3.org/ns/shacl#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
SCHEMA_TYPES = {kg.OWL + x for x in (
    "Class", "ObjectProperty", "DatatypeProperty", "AnnotationProperty", "Ontology", "Restriction",
    "TransitiveProperty", "SymmetricProperty", "FunctionalProperty", "InverseFunctionalProperty",
    "AsymmetricProperty", "ReflexiveProperty", "IrreflexiveProperty", "AllDisjointClasses", "AllDifferent",
    "OntologyProperty", "DeprecatedClass", "DeprecatedProperty")} | {
    RDF + "Property", RDFS + "Class", RDFS + "Datatype", RDFS + "ContainerMembershipProperty"} | {
    SH + x for x in ("NodeShape", "PropertyShape", "Shape", "SPARQLRule", "TripleRule", "SPARQLConstraint",
                     "SPARQLFunction", "SPARQLTarget", "SPARQLTargetType", "ConstraintComponent",
                     "PrefixDeclaration", "SPARQLAskValidator", "SPARQLSelectValidator", "Parameter",
                     "JSRule", "JSConstraint", "Function", "Rule", "Target", "TargetType")}


def vocab_ns(iri):
    """Namespace bucket of a vocabulary IRI, or None for instance data."""
    if iri.startswith(S223_BASE):
        # 223P vocab (standard223#, /1.0/model/, /1.0/inference/, /1.0/extensions/...) but never the
        # instance-data spaces (standard223/data/..., standard223/1.0/data/...)
        return None if "/data/" in iri[len(S223_BASE):] else "s223"
    if BRICK_RE.match(iri):
        return "brick"
    for p in VOCAB_PREFIXES:
        if iri.startswith(p):
            return p
    return None


def ns_of(iri):
    m = re.match(r"(.*[#/:])", iri)
    return m.group(1) if m else iri


def clean_triples(triples):
    """Returns (kept triples, stats)."""
    by_subj = collections.defaultdict(list)
    refs = collections.defaultdict(set)          # object node -> subjects referencing it
    for t in triples:
        s = t.subject.value
        by_subj[s].append(t)
        if isinstance(t.object, ox.NamedNode):
            refs[t.object.value].add(s)
    removed, why = set(), collections.Counter()
    ns_counts = collections.Counter()
    for s, ts in by_subj.items():
        if s.startswith("urn:bnode:"):
            continue
        ns = vocab_ns(s)
        if ns:
            removed.add(s); why["namespace"] += 1; ns_counts[ns] += len(ts)
            continue
        if any(t.predicate.value == kg.RDF_TYPE and t.object.value in SCHEMA_TYPES for t in ts):
            removed.add(s); why["schema_type"] += 1; ns_counts["typed:" + ns_of(s)] += len(ts)
    # typed blank nodes (shape / restriction bnodes)
    for s, ts in by_subj.items():
        if s.startswith("urn:bnode:") and any(t.predicate.value == kg.RDF_TYPE and t.object.value in SCHEMA_TYPES
                                              for t in ts):
            removed.add(s); why["schema_type_bnode"] += 1; ns_counts["bnode"] += len(ts)
    # blank nodes referenced only by removed subjects (fixpoint)
    changed = True
    while changed:
        changed = False
        for s in by_subj:
            if s in removed or not s.startswith("urn:bnode:"):
                continue
            r = refs.get(s)
            if r and r <= removed:
                removed.add(s); why["orphan_bnode"] += 1; ns_counts["bnode"] += len(by_subj[s])
                changed = True
    kept = [t for t in triples if t.subject.value not in removed]
    stats = {"triples_in": len(triples), "triples_out": len(kept), "triples_removed": len(triples) - len(kept),
             "subjects_removed": len(removed), "by_rule": dict(why),
             "triples_removed_by_ns": dict(ns_counts.most_common())}
    return kept, stats


def clean_file(src_nt, dst_nt):
    triples = list(ox.parse(path=str(src_nt), format=ox.RdfFormat.N_TRIPLES))
    kept, st = clean_triples(triples)
    dst_nt.parent.mkdir(parents=True, exist_ok=True)
    ox.serialize(kept, output=str(dst_nt), format=ox.RdfFormat.N_TRIPLES)
    return st


def main(do_bq=True, do_b2s=True, force=False, buildings=None):
    H.NT.mkdir(parents=True, exist_ok=True)
    rep_f = H.H / "clean_report.json"
    rep = json.loads(rep_f.read_text()) if rep_f.exists() else {"bq": {}, "b2s": {}}
    t0 = time.time()
    if do_bq:
        for b in H.BQ_BUILDINGS:
            if buildings and b not in buildings:
                continue
            dst = H.clean_nt("bq_" + b)
            if dst.exists() and not force and b in rep["bq"]:
                continue
            src = kg.nt_cache("bq_" + b, H.BQ_DIR / f"{b}.ttl")       # skolemized, read-only source
            rep["bq"][b] = clean_file(src, dst)
            s = rep["bq"][b]
            print(f"[bq] {b}: removed {s['triples_removed']}/{s['triples_in']} triples, {s['subjects_removed']} subjects "
                  f"{s['by_rule']}", flush=True)
    if do_b2s:
        for b in H.b2s_buildings():
            if buildings and b not in buildings:
                continue
            dst = H.clean_nt(b)
            if dst.exists() and not force and b in rep["b2s"]:
                continue
            rep["b2s"][b] = dict(clean_file(C2.nt_path(b), dst), split=C2.split_of(b), stratum=C2.stratum(b),
                                 graph_variant=C2.graph_variant(b))
        tot = collections.Counter()
        for b, s in rep["b2s"].items():
            tot["triples_in"] += s["triples_in"]; tot["triples_removed"] += s["triples_removed"]
            tot["buildings_touched"] += s["triples_removed"] > 0
        print(f"[b2s] {len(rep['b2s'])} buildings: {dict(tot)}", flush=True)
        for b, s in sorted(rep["b2s"].items(), key=lambda kv: -kv[1]["triples_removed"])[:15]:
            if s["triples_removed"]:
                print(f"   {b} [{s['stratum']}/{s['split']}]: -{s['triples_removed']}/{s['triples_in']} {s['by_rule']} "
                      f"{list(s['triples_removed_by_ns'].items())[:4]}", flush=True)
    rep_f.write_text(json.dumps(rep, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {rep_f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq", action="store_true", help="only BuildingQA graphs")
    ap.add_argument("--b2s", action="store_true", help="only Build2SPARQL graphs")
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    both = not (a.bq or a.b2s)
    main(a.bq or both, a.b2s or both, a.force, a.buildings)
