"""
Query decomposer for BKGQA (ontology-agnostic port of BrickQA/brick_decomposer.py).

Same structure and heuristics as BrickQueryDecomposer, but the LLM maps question
terms ONLY to classes/predicates from the vocabulary block of the loaded graph
(passed in), instead of a hard-coded Brick class table.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from vertexai.generative_models import GenerativeModel
    import vertexai
    VERTEXAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    VERTEXAI_AVAILABLE = False
    print("Warning: VertexAI not available. Decomposer will return empty decompositions.")

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class EntityMention:
    """A thing mentioned in the question, mapped to a vocabulary term of the graph."""
    mention: str          # original text from the question
    entity_id: str        # compact class/predicate/instance from the vocabulary (e.g. s223:TemperatureSensor)
    description: str      # what it is
    kind: str = "class"   # class | predicate | attribute | instance

    @property
    def sensor_id(self) -> str:  # BrickQA compatibility
        return self.entity_id


SensorMention = EntityMention  # BrickQA compatibility alias


@dataclass
class TemporalConstraint:
    has_constraint: bool
    type: str           # latest, recent_n, oldest_n, trend, range, specific, relative, none
    details: Dict
    sparql_pattern: str


@dataclass
class AggregationRequirement:
    required: bool
    operations: List[str]
    conditions: List[str]


@dataclass
class QueryDecomposition:
    entities: List[EntityMention]
    entity_count: int
    temporal: TemporalConstraint
    aggregation: AggregationRequirement
    query_intent: str
    raw_json: Dict
    output_attributes: List[str] = field(default_factory=list)

    # BrickQA-compatible names
    @property
    def sensors(self) -> List[EntityMention]:
        return self.entities

    @property
    def sensor_count(self) -> int:
        return self.entity_count


def empty_decomposition(question: str = "", reason: str = "") -> QueryDecomposition:
    return QueryDecomposition(
        entities=[], entity_count=0,
        temporal=TemporalConstraint(False, "none", {}, ""),
        aggregation=AggregationRequirement(False, [], []),
        query_intent=f"(no decomposition{': ' + reason if reason else ''})",
        raw_json={},
    )


class KGQueryDecomposer:
    """Decomposes questions into entities / temporal / aggregation / intent."""

    def __init__(self, model: str = "gemini-2.5-flash", project: str = None,
                 pre_call_sleep: float = 6.5, prompt_file: Optional[str] = None):
        self.model_name = model
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise RuntimeError("GCP project not provided and GOOGLE_CLOUD_PROJECT env var is not set.")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.pre_call_sleep = pre_call_sleep
        self.cooldown_seconds = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.prompt_file = Path(prompt_file) if prompt_file else PROMPT_DIR / "decomposition.prompt"

    # Same heuristic as BrickQueryDecomposer.should_decompose
    def should_decompose(self, question: str) -> bool:
        q = question.lower()

        def has_word(text: str, word: str) -> bool:
            return bool(re.search(r"\b" + re.escape(word) + r"\b", text))

        multi = [" and ", " or ", ",", "all ", "both ", "each "]
        comparison = ["compare", "versus", "vs", "between"]
        complex_kw = ["where", "which", "that", "outside", "exceed", "below", "above",
                      "difference", "delta", "change"]

        if len(question.split()) <= 6 and not any(k in q for k in multi):
            has_complexity = (any(k in q for k in multi) or any(has_word(q, k) for k in comparison)
                              or any(has_word(q, k) for k in complex_kw))
            if not has_complexity:
                return False
        if any(k in q for k in multi):
            return True
        if any(has_word(q, k) for k in comparison):
            return True
        if any(has_word(q, k) for k in complex_kw):
            return True
        if len(question.split()) > 12:
            return True
        return False

    def build_prompt(self, question: str, vocabulary_block: str = "", has_timeseries: bool = False,
                     context_block: Optional[str] = None) -> str:
        template = self.prompt_file.read_text(encoding="utf-8")
        ts_note = ("This graph CONTAINS timeseries observations (timestamps/values), so temporal constraints can apply."
                   if has_timeseries else
                   "This graph contains NO timeseries observations: set temporal.has_constraint to false and "
                   "temporal.type to \"none\" unless the question asks about a date-valued attribute.")
        ctx = f"\nAdditional building context:\n{context_block}\n" if context_block else ""
        return (template.replace("{{ question }}", question)
                .replace("{{ vocabulary }}", vocabulary_block or "(vocabulary not available)")
                .replace("{{ timeseries_note }}", ts_note)
                .replace("{{ context }}", ctx))

    def decompose(self, question: str, verbose: bool = True, vocabulary_block: str = "",
                  has_timeseries: bool = False, context_block: Optional[str] = None) -> QueryDecomposition:
        if not VERTEXAI_AVAILABLE:
            return empty_decomposition(question, "vertexai not installed")
        try:
            if self.pre_call_sleep:
                self.cooldown_seconds += self.pre_call_sleep
                time.sleep(self.pre_call_sleep)
            vertexai.init(project=self.project, location=self.location)
            prompt = self.build_prompt(question, vocabulary_block, has_timeseries, context_block)
            model = GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            um = getattr(response, "usage_metadata", None)
            if um:
                self.prompt_tokens += getattr(um, "prompt_token_count", 0) or 0
                self.completion_tokens += getattr(um, "candidates_token_count", 0) or 0
                self.total_tokens += getattr(um, "total_token_count", 0) or 0
            decomposition = self.parse_response(response.text, has_timeseries=has_timeseries)
            if verbose:
                self._print_decomposition(decomposition)
            return decomposition
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Decomposition failed: {e}")
            print("[INFO] Falling back to empty decomposition")
            return empty_decomposition(question, "decomposition failed")

    def parse_response(self, response_text: str, has_timeseries: bool = True) -> QueryDecomposition:
        text = response_text
        if "```json" in text:
            s = text.find("```json") + 7
            text = text[s:text.find("```", s)].strip()
        elif "```" in text:
            s = text.find("```") + 3
            text = text[s:text.find("```", s)].strip()
        else:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)
        return self._json_to_decomposition(json.loads(text), has_timeseries=has_timeseries)

    def _json_to_decomposition(self, d: Dict, has_timeseries: bool = True) -> QueryDecomposition:
        raw_entities = d.get("entities", d.get("sensors", [])) or []
        entities = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            entities.append(EntityMention(
                mention=str(e.get("mention", "")),
                entity_id=str(e.get("entity_id", e.get("sensor_id", ""))),
                description=str(e.get("description", "")),
                kind=str(e.get("kind", "class")),
            ))
        t = d.get("temporal", {}) or {}
        temporal = TemporalConstraint(
            has_constraint=bool(t.get("has_constraint", False)) and has_timeseries,
            type=t.get("type", "none") if has_timeseries else "none",
            details=t.get("details", {}) or {},
            sparql_pattern=t.get("sparql_pattern", "") or "",
        )
        a = d.get("aggregation", {}) or {}
        aggregation = AggregationRequirement(
            required=bool(a.get("required", False)),
            operations=[str(o).upper() for o in a.get("operations", []) or []],
            conditions=list(a.get("conditions", []) or []),
        )
        return QueryDecomposition(
            entities=entities,
            entity_count=int(d.get("entity_count", d.get("sensor_count", len(entities))) or len(entities)),
            temporal=temporal,
            aggregation=aggregation,
            query_intent=str(d.get("query_intent", "")),
            raw_json=d,
            output_attributes=[str(x) for x in d.get("output_attributes", []) or []],
        )

    def _print_decomposition(self, decomp: QueryDecomposition):
        print("\n" + "=" * 80)
        print("QUERY DECOMPOSITION")
        print("=" * 80)
        print(f"\nIntent: {decomp.query_intent}")
        print(f"\nEntities ({decomp.entity_count}):")
        for e in decomp.entities:
            print(f"  - {e.mention} -> {e.entity_id} [{e.kind}]  ({e.description})")
        if decomp.output_attributes:
            print(f"Output attributes: {', '.join(decomp.output_attributes)}")
        print(f"Temporal: {decomp.temporal.type if decomp.temporal.has_constraint else 'None'}")
        if decomp.temporal.has_constraint and decomp.temporal.details:
            print(f"    Details: {decomp.temporal.details}")
        if decomp.aggregation.required:
            print(f"Aggregation: {', '.join(decomp.aggregation.operations)}")
            if decomp.aggregation.conditions:
                print(f"    Conditions: {decomp.aggregation.conditions}")
        else:
            print("Aggregation: None")
        print("=" * 80 + "\n")
