"""
Temporal handler + validators for BKGQA (port of BrickQA/brick_temporal_handler.py).

Differences from BrickQA:
- Observation predicates come from the OntologyProfile (no hard-coded ref:/bldg:).
- The handler is a no-op unless the graph actually contains observation triples.
- If the decomposition gives no explicit SPARQL pattern, one is generated from the
  temporal type (BrickQA left the pattern empty, so nothing was applied).
- Patterns are only applied to queries that use a ?timestamp variable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .decomposer import QueryDecomposition
from .profile import OntologyProfile, local_name

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class TemporalApplication:
    original_sparql: str
    modified_sparql: str
    temporal_type: str
    pattern_applied: str
    was_modified: bool


class KGTemporalHandler:
    """Applies ORDER BY / FILTER / LIMIT temporal patterns (only when observations exist)."""

    def __init__(self, profile: Optional[OntologyProfile] = None, default_year: Optional[str] = None):
        self.profile = profile
        self.default_year = default_year

    @property
    def active(self) -> bool:
        return bool(self.profile is not None and self.profile.has_observations)

    def _noop(self, sparql: str) -> TemporalApplication:
        return TemporalApplication(sparql, sparql, "none", "", False)

    def apply_temporal_constraint(self, sparql: str, decomposition: Optional[QueryDecomposition] = None,
                                  temporal_type: Optional[str] = None,
                                  details: Optional[Dict] = None) -> TemporalApplication:
        if not self.active:
            return self._noop(sparql)
        if decomposition and decomposition.temporal.has_constraint:
            temporal_type = decomposition.temporal.type
            details = decomposition.temporal.details or {}
            pattern = decomposition.temporal.sparql_pattern or self._generate_pattern(temporal_type, details)
        elif temporal_type and details is not None:
            pattern = self._generate_pattern(temporal_type, details)
        else:
            return self._noop(sparql)
        if not pattern or "?timestamp" not in sparql:
            return TemporalApplication(sparql, sparql, temporal_type or "none", "", False)
        modified = self._apply_pattern(sparql, temporal_type, pattern, details or {})
        return TemporalApplication(sparql, modified, temporal_type, pattern, sparql != modified)

    @staticmethod
    def _dt(value: str) -> str:
        v = str(value).strip().rstrip("Z")
        if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$", v):
            if len(v) == 16:
                v += ":00"
            return f'"{v}"^^xsd:dateTime'
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return f'"{v}T00:00:00"^^xsd:dateTime'
        return f'"{v}"'

    def _generate_pattern(self, temporal_type: str, details: Dict) -> str:
        if temporal_type == "latest":
            return "ORDER BY DESC(?timestamp) LIMIT 1"
        if temporal_type == "recent_n":
            return f"ORDER BY DESC(?timestamp) LIMIT {int(details.get('limit', 10))}"
        if temporal_type == "oldest_n":
            return f"ORDER BY ?timestamp LIMIT {int(details.get('limit', 10))}"
        if temporal_type == "trend":
            return f"ORDER BY ?timestamp LIMIT {int(details.get('limit', 100))}"
        if temporal_type == "range":
            start, end = details.get("start_date", ""), details.get("end_date", "")
            if start and end:
                return f"FILTER(?timestamp >= {self._dt(start)} && ?timestamp < {self._dt(end)})"
            return ""
        if temporal_type == "specific":
            spec = details.get("specific_time", details.get("date", ""))
            return f'FILTER(CONTAINS(STR(?timestamp), "{spec}"))' if spec else ""
        return ""

    def _apply_pattern(self, sparql: str, temporal_type: str, pattern: str, details: Dict) -> str:
        if self._has_temporal_constraint(sparql, temporal_type):
            print(f"[INFO] Query already has {temporal_type} constraint, skipping")
            return sparql
        if "ORDER BY" in pattern:
            sparql = self._remove_existing_order_limit(sparql)
        if pattern.startswith("FILTER"):
            sparql = self._add_filter(sparql, pattern)
        elif "ORDER BY" in pattern:
            sparql = self._add_order_by(sparql, pattern)
        return sparql

    def _has_temporal_constraint(self, sparql: str, temporal_type: str) -> bool:
        if temporal_type in ["latest", "recent_n", "oldest_n", "trend"]:
            return "ORDER BY" in sparql.upper()
        if temporal_type in ["range", "specific"]:
            return bool(re.search(r"FILTER.*\?timestamp", sparql, re.IGNORECASE))
        return False

    def _remove_existing_order_limit(self, sparql: str) -> str:
        sparql = re.sub(r"\s*ORDER\s+BY\s+(?:DESC|ASC)?\s*\([^\)]+\)", "", sparql, flags=re.IGNORECASE)
        sparql = re.sub(r"\s*ORDER\s+BY\s+\S+", "", sparql, flags=re.IGNORECASE)
        sparql = re.sub(r"\s*LIMIT\s+\d+", "", sparql, flags=re.IGNORECASE)
        return sparql.strip()

    def _add_filter(self, sparql: str, filter_clause: str) -> str:
        m = re.search(r"WHERE\s*\{", sparql, re.IGNORECASE)
        if not m:
            print("[WARNING] No WHERE clause found, cannot add FILTER")
            return sparql
        pos, depth = m.end(), 1
        while pos < len(sparql) and depth > 0:
            if sparql[pos] == "{":
                depth += 1
            elif sparql[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0:
            print("[WARNING] Could not find matching brace for WHERE clause")
            return sparql
        ins = pos - 1
        return sparql[:ins] + f"\n  {filter_clause}\n" + sparql[ins:]

    def _add_order_by(self, sparql: str, order_pattern: str) -> str:
        result = sparql.rstrip()
        ob = re.search(r"ORDER BY\s+(?:DESC|ASC)?\s*\([^\)]+\)", order_pattern, re.IGNORECASE) or \
            re.search(r"ORDER BY\s+\S+", order_pattern, re.IGNORECASE)
        lim = re.search(r"LIMIT\s+\d+", order_pattern, re.IGNORECASE)
        if ob:
            result += f"\n{ob.group(0)}"
        if lim:
            result += f"\n{lim.group(0)}"
        return result

    def validate_temporal_query(self, sparql: str,
                                decomposition: Optional[QueryDecomposition] = None) -> Tuple[bool, str]:
        """Same checks as BrickQA's validate_temporal_query."""
        if not self.active:
            return True, "No timeseries observations in graph - temporal validation skipped"
        issues = []
        if decomposition and decomposition.temporal.has_constraint:
            tt = decomposition.temporal.type
            if tt in ["latest", "recent_n", "oldest_n", "trend"]:
                if not re.search(r"ORDER\s+BY.*\?timestamp", sparql, re.IGNORECASE):
                    issues.append(f"Expected ORDER BY for temporal type '{tt}' but not found")
                if not re.search(r"LIMIT\s+\d+", sparql, re.IGNORECASE):
                    issues.append(f"Expected LIMIT for temporal type '{tt}' but not found")
                if tt == "latest":
                    if not re.search(r"ORDER\s+BY\s+DESC", sparql, re.IGNORECASE):
                        issues.append("Expected DESC for 'latest' but not found")
                    if not re.search(r"LIMIT\s+1\b", sparql, re.IGNORECASE):
                        issues.append("Expected LIMIT 1 for 'latest' but not found")
                elif tt == "recent_n" and not re.search(r"ORDER\s+BY\s+DESC", sparql, re.IGNORECASE):
                    issues.append("Expected DESC for 'recent_n' but not found")
                elif tt == "oldest_n" and not re.search(r"ORDER\s+BY\s+(?!DESC)", sparql, re.IGNORECASE):
                    issues.append("Expected ASC (or no DESC) for 'oldest_n' but found DESC")
            elif tt in ["range", "specific"]:
                if not re.search(r"FILTER.*\?timestamp", sparql, re.IGNORECASE):
                    issues.append(f"Expected FILTER on ?timestamp for temporal type '{tt}' but not found")
                if tt == "range" and not re.search(r"FILTER.*\?timestamp.*[<>=]", sparql, re.IGNORECASE):
                    issues.append("Expected comparison operators (>=, <=) for 'range' but not found")
        else:
            if "?timestamp" in sparql.lower():
                has_order = re.search(r"ORDER\s+BY.*\?timestamp", sparql, re.IGNORECASE)
                has_filter = re.search(r"FILTER.*\?timestamp", sparql, re.IGNORECASE)
                if not has_order and not has_filter:
                    issues.append("Query has ?timestamp variable but no ORDER BY or FILTER on it")
            if re.search(r"ORDER\s+BY", sparql, re.IGNORECASE) and not re.search(r"LIMIT\s+\d+", sparql, re.IGNORECASE):
                issues.append("Query has ORDER BY but no LIMIT (may return too many results)")
        if issues:
            return False, "; ".join(issues)
        return True, "Temporal constraints are valid"

    def render_guide(self, date_range: Tuple[Optional[str], Optional[str]] = (None, None)) -> str:
        """Temporal patterns guide with this graph's observation predicates filled in."""
        path = PROMPT_DIR / "temporal_patterns.txt"
        text = path.read_text(encoding="utf-8")
        p = self.profile
        if p is not None and p.has_observations:
            obs = p.obs_compact("observation") or "?observationLink"
            val = p.obs_compact("value")
            pattern = f"?point {obs} ?obs . ?obs {p.obs_compact('timestamp')} ?timestamp" + \
                      (f" ; {val} ?value ." if val else " .")
        else:
            pattern = "(no observation triples found in this graph)"
        dr = f"{date_range[0]} to {date_range[1]}" if date_range and date_range[0] else "an unknown date range"
        return (text.replace("{{ observation_pattern }}", pattern)
                .replace("{{ date_range }}", dr)
                .replace("{{ default_year }}", str(self.default_year or "of the data")))


# --------------------------------------------------------------------------
# Validators (warn-only, as in BrickQA)
# --------------------------------------------------------------------------
def _mentions(sparql: str, entity_id: str, profile: Optional[OntologyProfile]) -> bool:
    if not entity_id:
        return True
    eid = entity_id.strip()
    if eid in sparql:
        return True
    if profile is not None:
        full = profile.expand(eid)
        if full and (f"<{full}>" in sparql or profile.compact(full) in sparql):
            return True
    loc = local_name(eid.split(":", 1)[-1])
    return bool(loc) and re.search(r"[:/#<]" + re.escape(loc) + r"\b", sparql) is not None


def validate_entity_coverage(sparql: str, decomposition: QueryDecomposition,
                             profile: Optional[OntologyProfile] = None) -> Tuple[bool, str]:
    """All class/predicate terms from the decomposition appear in the SPARQL (instances skipped)."""
    ents = [e for e in decomposition.entities if e.kind != "instance" and e.entity_id]
    if not ents:
        return True, "No entities in decomposition"
    missing = sorted({e.entity_id for e in ents if not _mentions(sparql, e.entity_id, profile)})
    if missing:
        return False, f"Missing entities: {', '.join(missing)}"
    return True, f"All {len(ents)} entities present"


validate_sensor_coverage = validate_entity_coverage  # BrickQA name


def validate_aggregation(sparql: str, decomposition: QueryDecomposition) -> Tuple[bool, str]:
    if not decomposition.aggregation.required:
        return True, "No aggregation required"
    up = sparql.upper()
    missing = [op for op in decomposition.aggregation.operations if f"{op}(" not in up.replace(" (", "(")]
    if missing:
        issues = [f"Missing aggregation operations: {', '.join(missing)}"]
        if "GROUP BY" not in up:
            issues.append("Missing GROUP BY clause")
        return False, "; ".join(issues)
    return True, f"All aggregations present: {', '.join(decomposition.aggregation.operations)}"


def validate_multi_sensor_joins(sparql: str, decomposition: QueryDecomposition,
                                profile: Optional[OntologyProfile] = None) -> Tuple[bool, str]:
    """Multi-entity timeseries queries should join on one ?timestamp (only if graph has observations)."""
    if profile is None or not profile.has_observations:
        return True, "No timeseries observations - join validation skipped"
    if "?timestamp" not in sparql.lower():
        return True, "Query does not use observations - join validation skipped"
    ents = [e for e in decomposition.entities if e.entity_id]
    if len(ents) <= 1:
        return True, "Single entity query - no joins needed"
    ts_full = profile.obs_predicates["timestamp"]
    ts_alts = [re.escape(profile.compact(ts_full)), re.escape(f"<{ts_full}>")]
    ts_re = "(?:" + "|".join(ts_alts) + r")\s+\?timestamp"
    joined = sum(1 for e in ents
                 if re.search(re.escape(e.entity_id) + r".*?" + ts_re, sparql, re.DOTALL | re.IGNORECASE))
    issues = []
    if 0 < joined < len(ents):
        issues.append(f"Multi-entity query: {len(ents) - joined} entity(ies) not joined on ?timestamp "
                      f"(found {joined}/{len(ents)} with timestamp join)")
    obs_vars = set(re.findall(r"\?(\w+)_obs\b", sparql)) | set(re.findall(r"\?obs_(\w+)\b", sparql))
    if len(obs_vars) < len(ents) and re.search(r"\?obs\b", sparql):
        issues.append("Multi-entity query should use unique observation variables per quantity instead of reusing ?obs")
    if issues:
        return False, "; ".join(issues)
    return True, f"Multi-entity timestamp joins valid ({len(ents)} entities)"


def should_use_temporal_guide(decomposition: QueryDecomposition) -> bool:
    """Same policy as BrickQA: simple patterns -> handler; complex/unknown -> guide."""
    if not decomposition or not decomposition.temporal.has_constraint:
        return False
    t = decomposition.temporal.type
    if t in ["latest", "recent_n", "oldest_n", "trend"]:
        return False
    return True
