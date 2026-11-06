"""
OntologyProfile: everything the agent needs to know about the *vocabulary* of a
loaded building knowledge graph, computed from the graph itself.

- prefix map (TTL declarations + rdflib namespaces + standard ones)
- robust IRI compaction ('#' and '/' namespaces, <full-iri> fallback)
- ontology detection (brick / s223 / hybrid / generic)
- vocabulary summary (classes, object predicates, literal attributes)
- short, standards-based pattern guidance per ontology
- detection of timeseries observation triples (for the temporal handler)

Nothing here is specific to one building, and nothing is hard-coded to Brick.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

# --------------------------------------------------------------------------
# Standard namespaces
# --------------------------------------------------------------------------
BRICK_NS = "https://brickschema.org/schema/Brick#"
REF_NS = "https://brickschema.org/schema/Brick/ref#"
S223_NS = "http://data.ashrae.org/standard223#"
SH_NS = "http://www.w3.org/ns/shacl#"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"

STANDARD_PREFIXES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "brick": BRICK_NS,
    "ref": REF_NS,
    "s223": S223_NS,
    "qudt": "http://qudt.org/schema/qudt/",
    "unit": "http://qudt.org/vocab/unit/",
    "quantitykind": "http://qudt.org/vocab/quantitykind/",
}
CORE_PREFIXES = ("rdf", "rdfs", "owl", "xsd")
ONTOLOGY_PREFIXES = {
    "brick": ("brick", "ref"),
    "s223": ("s223", "qudt", "unit", "quantitykind"),
    "hybrid": ("brick", "ref", "s223", "qudt", "unit", "quantitykind"),
    "generic": (),
}

# Classes whose instances are ontology/schema definitions, not building data
META_CLASSES = {
    str(RDFS.Class), str(OWL.Class), str(RDF.Property), str(OWL.ObjectProperty),
    str(OWL.DatatypeProperty), str(OWL.AnnotationProperty), str(OWL.Ontology),
    str(OWL.Restriction), str(OWL.TransitiveProperty), str(OWL.SymmetricProperty),
    str(OWL.FunctionalProperty), str(OWL.InverseFunctionalProperty),
    str(RDFS.Datatype),
    SH_NS + "NodeShape", SH_NS + "PropertyShape", SH_NS + "SPARQLRule",
    SH_NS + "TripleRule", SH_NS + "SPARQLConstraint", SH_NS + "Parameter",
    S223_NS + "Class", BRICK_NS + "EntityProperty",
}
# Predicates that only make sense in schema definitions
SCHEMA_PREDICATES = {
    str(RDFS.subClassOf), str(RDFS.subPropertyOf), str(RDFS.domain), str(RDFS.range),
    str(OWL.equivalentClass), str(OWL.inverseOf), str(OWL.imports), str(OWL.versionInfo),
    str(RDF.first), str(RDF.rest),
}
# Subjects in these namespaces are standard vocabulary definitions, not building instances
REFERENCE_NAMESPACES = (
    "https://brickschema.org/schema/", "http://data.ashrae.org/standard223#",
    "http://qudt.org/", "https://qudt.org/", "http://www.w3.org/", "https://w3id.org/rec#",
    "http://www.linkedmodel.org/", "http://voag.linkedmodel.org/", "https://si-digital-framework.org/",
    "http://purl.org/dc/", "http://www.opengis.net/",
)
LABEL_PREDICATES = {
    str(RDFS.label), SKOS_NS + "prefLabel", SKOS_NS + "altLabel",
    "http://schema.org/name", "https://schema.org/name",
}

_PREFIX_LINE_RE = re.compile(
    r"^\s*(?:@prefix|PREFIX)\s+([A-Za-z][\w\-.]*)?:\s*<([^>\s]*)>", re.IGNORECASE | re.MULTILINE
)
_PN_LOCAL_RE = re.compile(r"^(?:[A-Za-z0-9_](?:[A-Za-z0-9_\-.]*[A-Za-z0-9_\-])?)?$")


def parse_ttl_prefixes(ttl_path: str, max_bytes: int = 2_000_000) -> Dict[str, str]:
    """Read @prefix / PREFIX declarations from a Turtle file (order preserved)."""
    prefixes: Dict[str, str] = {}
    try:
        with open(ttl_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_bytes)
    except OSError:
        return prefixes
    for m in _PREFIX_LINE_RE.finditer(text):
        name = m.group(1) or ""
        if name not in prefixes:
            prefixes[name] = m.group(2)
    return prefixes


def local_name(iri: str) -> str:
    """Local name of an IRI for both '#' and '/' namespaces."""
    iri = str(iri)
    for sep in ("#", "/", ":"):
        idx = iri.rfind(sep)
        if idx != -1 and idx < len(iri) - 1:
            return iri[idx + 1:]
    return iri


def split_words(text: str) -> List[str]:
    """'Zone_Air_TemperatureSensor-1' -> ['zone','air','temperature','sensor','1']"""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return [w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if w]


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------
@dataclass
class OntologyProfile:
    prefixes: Dict[str, str]                       # prefix -> namespace (priority-ordered)
    declared_prefixes: Dict[str, str]              # declared in the TTL file
    ontology: str = "generic"                      # brick | s223 | hybrid | generic
    ns_scores: Dict[str, int] = field(default_factory=dict)
    used_prefixes: set = field(default_factory=set)
    class_counts: Counter = field(default_factory=Counter)
    obj_pred_counts: Counter = field(default_factory=Counter)
    obj_pred_pairs: Dict[str, Counter] = field(default_factory=dict)
    attr_pred_counts: Counter = field(default_factory=Counter)
    attr_pred_classes: Dict[str, Counter] = field(default_factory=dict)
    attr_examples: Dict[str, str] = field(default_factory=dict)
    class_attr_sets: Dict[str, Counter] = field(default_factory=dict)
    class_cotypes: Dict[str, Counter] = field(default_factory=dict)
    has_class_hierarchy: bool = False
    n_triples: int = 0
    n_instance_triples: int = 0
    # timeseries observation predicates (None if graph has no observations)
    obs_predicates: Dict[str, Optional[str]] = field(default_factory=dict)
    instance_namespace: Optional[str] = None
    _ns_sorted: List[Tuple[str, str]] = field(default_factory=list, repr=False)
    _compact_cache: Dict[str, str] = field(default_factory=dict, repr=False)

    # ---------------- prefixes / compaction ----------------
    def _rebuild_ns_index(self):
        seen = set()
        pairs = []
        for p, ns in self.prefixes.items():  # dict order == priority order
            if ns in seen or not ns:
                continue
            seen.add(ns)
            pairs.append((ns, p))
        self._ns_sorted = sorted(pairs, key=lambda x: -len(x[0]))
        self._compact_cache = {}

    def compact(self, iri) -> str:
        """Compact an IRI with the prefix map; fall back to <full-iri>."""
        if isinstance(iri, BNode):
            return "[blank node]"
        if isinstance(iri, Literal):
            return format_literal(iri)
        s = str(iri)
        hit = self._compact_cache.get(s)
        if hit is not None:
            return hit
        out = f"<{s}>"
        for ns, p in self._ns_sorted:
            if s.startswith(ns):
                loc = s[len(ns):]
                if _PN_LOCAL_RE.match(loc):
                    out = f"{p}:{loc}"
                    break
        if len(self._compact_cache) < 500_000:
            self._compact_cache[s] = out
        return out

    def expand(self, term: str) -> Optional[str]:
        """'s223:Sensor' or '<iri>' or 'http://...' -> full IRI (None if not resolvable)."""
        term = term.strip()
        if term.startswith("<") and term.endswith(">"):
            return term[1:-1]
        if re.match(r"^[a-z][a-z0-9+.\-]*://", term, re.IGNORECASE) or term.startswith("urn:"):
            return term
        if ":" in term:
            p, loc = term.split(":", 1)
            if p in self.prefixes:
                return self.prefixes[p] + loc
        return None

    def prefix_declarations(self, names=None) -> str:
        names = list(names) if names is not None else list(self.prefixes)
        return "\n".join(f"PREFIX {n}: <{self.prefixes[n]}>" for n in names if n in self.prefixes)

    def prompt_prefix_names(self) -> List[str]:
        """Prefixes shown in the prompt: declared in the file + core + ontology-relevant + used."""
        names: List[str] = []
        for n in list(self.declared_prefixes) + list(CORE_PREFIXES) + list(ONTOLOGY_PREFIXES.get(self.ontology, ())):
            if n and n in self.prefixes and n not in names:
                names.append(n)
        for n in sorted(self.used_prefixes):
            if n and n in self.prefixes and n not in names:
                names.append(n)
        return names

    def prefix_block(self) -> str:
        return self.prefix_declarations(self.prompt_prefix_names())

    # ---------------- observations ----------------
    @property
    def has_observations(self) -> bool:
        return bool(self.obs_predicates.get("timestamp"))

    def obs_compact(self, key: str) -> Optional[str]:
        iri = self.obs_predicates.get(key)
        return self.compact(iri) if iri else None

    # ---------------- vocabulary summary ----------------
    def vocabulary_block(self, max_chars: int = 6000) -> str:
        # (classes, relationships, attributes, attribute-sets, pairs per relationship),
        # tried in order until the block fits; relationships are trimmed last.
        configs = [
            (40, 30, 20, 12, 3), (40, 30, 20, 10, 2), (30, 30, 16, 8, 2), (25, 25, 14, 6, 2),
            (20, 22, 12, 6, 1), (15, 18, 10, 5, 1), (12, 14, 8, 4, 1), (10, 10, 6, 3, 1), (6, 6, 4, 2, 1),
        ]
        text = ""
        for cfg in configs:
            text = self._render_vocab(*cfg)
            if len(text) <= max_chars:
                break
        if len(text) > max_chars:
            text = text[: max_chars - 20].rsplit("\n", 1)[0] + "\n  ... (truncated)"
        return text

    def _render_vocab(self, n_cls, n_rel, n_attr, n_sets, n_pairs) -> str:
        c = self.compact
        lines = [f"Vocabulary actually present in this graph (computed from {self.n_instance_triples} "
                 f"instance-data triples; ontology: {self.ontology}):"]
        # classes
        cls = self.class_counts.most_common(n_cls)
        lines.append(f"Classes used (rdf:type objects, instance count; {len(self.class_counts)} total, top {len(cls)}):")
        lines.append("  " + ", ".join(f"{c(k)} ({v})" for k, v in cls) if cls else "  (none)")
        lines.append("  rdfs:subClassOf axioms for these classes are " +
                     ("PRESENT in the graph (rdf:type/rdfs:subClassOf* works)." if self.has_class_hierarchy
                      else "NOT in the graph (match the exact classes listed; use VALUES for alternatives)."))
        # relationships
        rels = self.obj_pred_counts.most_common(n_rel)
        lines.append(f"Relationships (object predicates, count: typical subject-class -> object-class; "
                     f"{len(self.obj_pred_counts)} total):")
        for p, n in rels:
            pairs = self.obj_pred_pairs.get(p, Counter()).most_common(n_pairs)
            pstr = "; ".join(f"{a} -> {b} ({k})" for (a, b), k in pairs)
            lines.append(f"  {c(p)} ({n}): {pstr}")
        # attributes
        attrs = self.attr_pred_counts.most_common(n_attr)
        lines.append(f"Attributes (literal-valued predicates, count: carried by; example; "
                     f"{len(self.attr_pred_counts)} total):")
        for p, n in attrs:
            carriers = ", ".join(k for k, _ in self.attr_pred_classes.get(p, Counter()).most_common(3))
            ex = self.attr_examples.get(p, "")
            lines.append(f"  {c(p)} ({n}): on {carriers}; e.g. {ex}")
        # attribute sets by class
        sets = sorted(self.class_attr_sets.items(), key=lambda kv: -sum(kv[1].values()))[:n_sets]
        if sets:
            lines.append("Attribute sets by class (which literal attributes each kind of node carries):")
            for k, cnt in sets:
                co = [t for t, m in self.class_cotypes.get(k, Counter()).most_common(2)]
                cotxt = f" [also typed {', '.join(co)}]" if co else ""
                lines.append(f"  {k}{cotxt} has attributes " + ", ".join(c(p) for p, _ in cnt.most_common(8)))
        if self.has_observations:
            o = self.obs_predicates
            lines.append("Timeseries observations present: ?point "
                         f"{c(o['observation']) if o.get('observation') else '?p'} ?obs . ?obs "
                         f"{c(o['timestamp'])} ?timestamp" +
                         (f" ; {c(o['value'])} ?value ." if o.get("value") else " ."))
        return "\n".join(lines)

    # ---------------- guidance ----------------
    def pattern_guidance(self) -> str:
        parts = []
        if self.ontology in ("brick", "hybrid"):
            parts.append(BRICK_GUIDANCE)
        if self.ontology in ("s223", "hybrid"):
            parts.append(S223_GUIDANCE)
        parts.append(GENERIC_GUIDANCE)
        if self.ontology == "hybrid":
            parts.insert(0, "This graph mixes Brick and ASHRAE 223P terms; an entity may carry both kinds "
                            "of types/relations. Check the vocabulary block for which applies.")
        return "\n\n".join(parts)


def format_literal(lit: Literal, max_len: int = 60) -> str:
    s = str(lit)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    s = s.replace("\n", " ").replace('"', '\\"')
    return f'"{s}"'


# --------------------------------------------------------------------------
# Guidance text (from general knowledge of the standards only)
# --------------------------------------------------------------------------
BRICK_GUIDANCE = """Brick patterns (use only predicates that appear in the vocabulary block):
- Points of equipment: ?equip brick:hasPoint ?point  (inverse: ?point brick:isPointOf ?equip).
- Composition: brick:hasPart / brick:isPartOf (building -> floor -> space; zone -> spaces; equipment -> sub-equipment).
- Location: brick:hasLocation / brick:isLocationOf (equipment/points located in spaces).
- Flow/feeds: brick:feeds / brick:isFedBy (e.g. AHU -> VAV -> zone); brick:feeds+ for chains.
- Graphs often contain only one direction of an inverse pair: use alternation, e.g. (brick:hasPoint|^brick:isPointOf).
- External data: ?point ref:hasExternalReference ?r . ?r ref:hasTimeseriesId ?id (also ref:storedAt, IFC refs
  with ref:ifcName / ref:ifcGlobalID, BACnet refs). External-reference nodes are often blank nodes.
- Static values: entity properties such as brick:area [ brick:value ?v ; brick:hasUnit ?u ]."""

S223_GUIDANCE = """ASHRAE 223P patterns (use only predicates that appear in the vocabulary block):
- Properties: ?thing s223:hasProperty ?prop ; ?prop qudt:hasQuantityKind ?qk ; qudt:hasUnit ?unit ;
  s223:hasAspect / s223:ofSubstance / s223:ofMedium qualify it; static values via s223:hasValue.
- Sensors: ?sensor s223:observes ?prop ; s223:hasObservationLocation ?loc (connection point, connection or space);
  s223:hasPhysicalLocation ?space. Actuators: s223:actuates ?equip ; s223:commandedByProperty ?prop.
- Composition: ?equip s223:contains ?sub (equipment within equipment / systems).
- Topology: ?equip s223:hasConnectionPoint ?cp (s223:InletConnectionPoint / s223:OutletConnectionPoint);
  ?cp s223:isConnectionPointOf ?equip ; ?cp s223:cnx ?connection ; ?equip s223:connectsThrough ?connection ;
  ?a s223:connectedTo ?b (may be absent if not inferred). Connections/points carry s223:hasMedium.
- Spaces/zones: ?physicalSpace s223:encloses ?domainSpace ; ?zone s223:hasDomainSpace ?domainSpace ;
  s223:hasDomain gives the domain (e.g. HVAC, lighting).
- External data: ?prop s223:hasExternalReference ?ref ; the reference node (s223:ExternalReference or a subclass)
  carries the identifiers/endpoints as literal attributes - see the attribute sets in the vocabulary block."""

GENERIC_GUIDANCE = """General patterns:
- Find entities by class: ?x rdf:type <Class> (add /rdfs:subClassOf* only if subclass axioms are present).
- Names/identifiers are literal attributes (rdfs:label, ...): project them explicitly when asked.
- Prefer starting from a class or relationship listed in the vocabulary block, then follow relationships
  step by step, verifying each hop with execute_sparql."""


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
def build_profile(graph: Graph, ttl_path: Optional[str] = None,
                  extra_prefixes: Optional[Dict[str, str]] = None) -> OntologyProfile:
    declared: Dict[str, str] = {}
    if ttl_path:
        declared = parse_ttl_prefixes(ttl_path)
    if extra_prefixes:
        for k, v in extra_prefixes.items():
            declared.setdefault(k, v)

    # Priority: declared > standard > rdflib bindings (name collisions keep the higher priority)
    prefixes: Dict[str, str] = {}
    for k, v in declared.items():
        if k:  # skip the empty prefix in the map (cannot be used safely in queries)
            prefixes[k] = v
    for k, v in STANDARD_PREFIXES.items():
        prefixes.setdefault(k, v)
    for k, v in graph.namespaces():
        k = str(k)
        if k and k not in prefixes and re.match(r"^[A-Za-z][\w\-]*$", k):
            prefixes[k] = str(v)

    prof = OntologyProfile(prefixes=prefixes, declared_prefixes={k: v for k, v in declared.items() if k})
    prof._rebuild_ns_index()
    _compute_stats(graph, prof)
    return prof


def _ns_of(iri: str) -> str:
    for sep in ("#", "/"):
        idx = iri.rfind(sep)
        if idx != -1:
            return iri[: idx + 1]
    return iri


def _compute_stats(graph: Graph, prof: OntologyProfile):
    c = prof.compact
    rdf_type = RDF.type
    types: Dict[object, List[str]] = defaultdict(list)
    schema_subjects = set()
    subclass_subjects = set()
    sh_prefix = SH_NS

    triples = list(graph.triples((None, None, None)))
    prof.n_triples = len(triples)

    sub_props = []
    for s, p, o in triples:
        if p == rdf_type:
            types[s].append(str(o))
            if str(o) in META_CLASSES:
                schema_subjects.add(s)
        else:
            ps = str(p)
            if ps.startswith(sh_prefix):
                schema_subjects.add(s)
            elif ps == str(RDFS.subClassOf):
                subclass_subjects.add(s)
            elif ps == str(RDFS.subPropertyOf):
                sub_props.append((str(s), str(o)))
        # definitions living in a standard's own namespace (ontology terms, QUDT units, tags, ...)
        if isinstance(s, URIRef) and str(s).startswith(REFERENCE_NAMESPACES):
            schema_subjects.add(s)
    schema_subjects |= subclass_subjects

    # blank nodes hanging off schema definitions (lists, restrictions, metadata) are schema too
    bnode_parent: Dict[object, object] = {}
    for s, p, o in triples:
        if isinstance(o, BNode) and o not in bnode_parent:
            bnode_parent[o] = s
    for _ in range(10):
        added = [b for b, par in bnode_parent.items() if b not in schema_subjects and par in schema_subjects]
        if not added:
            break
        schema_subjects.update(added)

    # class counts over instance subjects
    class_counts: Counter = Counter()
    for s, ts in types.items():
        if s in schema_subjects:
            continue
        for t in set(ts):
            class_counts[t] += 1
    prof.class_counts = class_counts
    prof.has_class_hierarchy = any(URIRef(k) in subclass_subjects for k, _ in class_counts.most_common(50))

    # incoming predicate for untyped blank nodes
    bnode_in: Dict[object, str] = {}
    for s, p, o in triples:
        if isinstance(o, BNode) and o not in bnode_in and p != rdf_type:
            bnode_in[o] = str(p)

    primary_cache: Dict[object, str] = {}

    def primary(node) -> str:
        hit = primary_cache.get(node)
        if hit is not None:
            return hit
        ts = types.get(node)
        if ts:
            best = min(set(ts), key=lambda t: (class_counts.get(t, 10**9), -len(t)))
            out = c(best)
        elif isinstance(node, BNode):
            via = bnode_in.get(node)
            out = f"[blank node via {c(via)}]" if via else "[blank node]"
        elif isinstance(node, Literal):
            out = "literal"
        else:
            out = c(node)  # untyped IRI (e.g. an enumeration value or unit)
        primary_cache[node] = out
        return out

    obj_counts: Counter = Counter()
    obj_pairs: Dict[str, Counter] = defaultdict(Counter)
    attr_counts: Counter = Counter()
    attr_classes: Dict[str, Counter] = defaultdict(Counter)
    attr_examples: Dict[str, str] = {}
    class_attr: Dict[str, Counter] = defaultdict(Counter)
    class_cotypes: Dict[str, Counter] = defaultdict(Counter)
    ns_pred = Counter()
    ns_class = Counter()
    used_ns = Counter()
    n_inst = 0

    for s, ts in types.items():
        if s in schema_subjects:
            continue
        k = primary(s)
        for t in set(ts):
            ct = c(t)
            if ct != k:
                class_cotypes[k][ct] += 1
            ns_class[_ns_of(t)] += 1
            used_ns[_ns_of(t)] += 1
        if isinstance(s, URIRef):
            used_ns[_ns_of(str(s))] += 1

    for s, p, o in triples:
        if s in schema_subjects or p == rdf_type:
            continue
        ps = str(p)
        if ps in SCHEMA_PREDICATES or ps.startswith(sh_prefix):
            continue
        n_inst += 1
        ns_pred[_ns_of(ps)] += 1
        used_ns[_ns_of(ps)] += 1
        if isinstance(o, Literal):
            attr_counts[ps] += 1
            k = primary(s)
            attr_classes[ps][k] += 1
            class_attr[k][ps] += 1
            if ps not in attr_examples and str(o).strip():
                attr_examples[ps] = format_literal(o, 40)
            if o.datatype is not None:
                used_ns[_ns_of(str(o.datatype))] += 1
        else:
            obj_counts[ps] += 1
            obj_pairs[ps][(primary(s), primary(o))] += 1

    # drop super-properties (e.g. a root 'Relationship' property) when their sub-properties are used
    used_preds = set(obj_counts) | set(attr_counts)
    supers = {sup for sub, sup in sub_props if sub in used_preds and sub != sup}
    for sup in supers:
        n_sup = obj_counts.pop(sup, 0) + attr_counts.pop(sup, 0)
        ns_pred[_ns_of(sup)] -= n_sup  # inferred super-property triples do not count for detection
        n_inst -= n_sup
        obj_pairs.pop(sup, None)
        attr_classes.pop(sup, None)
        for cnt in class_attr.values():
            cnt.pop(sup, None)

    prof.n_instance_triples = n_inst
    prof.obj_pred_counts = obj_counts
    prof.obj_pred_pairs = dict(obj_pairs)
    prof.attr_pred_counts = attr_counts
    prof.attr_pred_classes = dict(attr_classes)
    prof.attr_examples = attr_examples
    prof.class_attr_sets = dict(class_attr)
    # keep only co-types seen on >= 50% of the class's instances
    prof.class_cotypes = _filter_cotypes(class_cotypes, types, schema_subjects, primary)

    # used prefixes
    ns_to_prefix = {ns: p for ns, p in prof._ns_sorted}
    prof.used_prefixes = {ns_to_prefix[ns] for ns in used_ns if ns in ns_to_prefix}

    # ontology detection: frequency of Brick vs 223P namespaces in predicates + classes
    brick = ns_pred[BRICK_NS] + ns_class[BRICK_NS]
    s223 = ns_pred[S223_NS] + ns_class[S223_NS]
    prof.ns_scores = {"brick": brick, "s223": s223}
    if brick and s223 and min(brick, s223) >= max(50, 0.05 * max(brick, s223)):
        prof.ontology = "hybrid"
    elif brick > s223 and brick > 0:
        prof.ontology = "brick"
    elif s223 > 0:
        prof.ontology = "s223"
    else:
        prof.ontology = "generic"

    # instance namespace (most common namespace of typed instance IRIs)
    inst_ns = Counter(_ns_of(str(s)) for s in types if isinstance(s, URIRef) and s not in schema_subjects)
    prof.instance_namespace = inst_ns.most_common(1)[0][0] if inst_ns else None

    # timeseries observations: a literal-valued predicate named hasTimestamp/timestamp
    obs = {"observation": None, "timestamp": None, "value": None}
    for p in attr_counts:
        if local_name(p).lower() in ("hastimestamp", "timestamp"):
            obs["timestamp"] = p
            ns = _ns_of(p)
            for cand in attr_counts:
                if _ns_of(cand) == ns and local_name(cand).lower() in ("hasvalue", "value"):
                    obs["value"] = cand
            for cand in obj_counts:
                if local_name(cand).lower() in ("hasobservation", "observation"):
                    obs["observation"] = cand
            break
    prof.obs_predicates = obs if obs["timestamp"] else {}


def _filter_cotypes(class_cotypes, types, schema_subjects, primary) -> Dict[str, Counter]:
    totals = Counter()
    for s in types:
        if s not in schema_subjects:
            totals[primary(s)] += 1
    out = {}
    for k, cnt in class_cotypes.items():
        keep = Counter({t: n for t, n in cnt.items() if n >= 0.5 * totals.get(k, 1)})
        if keep:
            out[k] = keep
    return out
