"""Source-backed records for the optional metrics module (ISO date strings)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MetricDefinition:
    code: str
    name: str
    unit: str
    better: str = "none"
    aliases: list[str] = field(default_factory=list)
    normalize_by: str | None = None
    aggregation: str = "mean"
    confirmed_by_human: bool = True


@dataclass
class MetricSource:
    file: str
    quote: str
    sheet: str | None = None
    cell: str | None = None
    page: int | None = None
    locator: str = ""
    content_hash: str = ""


@dataclass
class MetricValue:
    id: str
    metric: str
    value: float
    unit_scope: str
    period_start: str
    period_end: str
    granularity: str
    source: MetricSource
    is_synthetic: bool = False
    extracted_by: str = "rule"
    confirmed_by_human: bool = False
    unit: str = ""


@dataclass
class MetricAmbiguity:
    id: str
    reason: str
    source: MetricSource
    proposed: dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False


@dataclass
class IngestionResult:
    values: list[MetricValue] = field(default_factory=list)
    ambiguous: list[MetricAmbiguity] = field(default_factory=list)
    custom_definitions: list[MetricDefinition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StructuralEvent:
    id: str
    type: str
    units: list[str]
    effective_date: str
    evidence: list[dict[str, Any]]
    finding_id: str = ""
    description: str = ""
    metric_codes: list[str] = field(default_factory=list)
    mechanism: str = ""
    mechanism_confirmed: bool = False
    is_synthetic: bool = False
    date_basis: str = "user_confirmed"


@dataclass
class PeriodComparison:
    id: str
    metric: str
    unit_scope: str
    event_id: str
    before: dict[str, Any]
    after: dict[str, Any]
    delta_abs: float
    delta_pct: float | None
    normalized_delta: float | None = None
    is_significant: str = "insufficient_data"
    method: str = "simple"
    metric_ids: list[str] = field(default_factory=list)
    control_scope: str | None = None
    adjusted_delta: float | None = None
    aggregation: str = "mean"
    excluded_periods: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    normalization_sources: list[str] = field(default_factory=list)


@dataclass
class Attribution:
    id: str
    event_id: str
    comparison_id: str
    claim: str
    mechanism: str
    confidence: str
    confidence_reasons: list[str]
    alternative_explanations: list[str]
    evidence: dict[str, Any]
    opponent_outcome: str = "weakened"
    opponent_reasoning: str = ""
    is_synthetic: bool = False
    method: str = "rule"


@dataclass
class Hypothesis:
    id: str
    source_type: str
    source_id: str
    prediction: str
    metric: str
    expected_direction: str
    check_after: str
    unit_scope: str = "BVA"
    event_id: str = ""
    status: str = "pending"
    checked_with: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    explanation: str = ""


@dataclass
class KnowledgeRecord:
    id: str
    change_pattern: str
    context: dict[str, Any]
    observed_effect: dict[str, Any]
    confidence: str
    cases: list[str]
    n_cases: int
    is_synthetic: bool = False
    diversity: int = 1


@dataclass
class Recommendation:
    id: str
    action: str
    type: str
    based_on: dict[str, Any]
    expected_effect: dict[str, Any]
    confidence: str
    risks: list[str]
    target_units: list[str] = field(default_factory=list)
    redline_id: str | None = None
    requires_human_decision: bool = True
    decision: str = "discuss"
    is_synthetic: bool = False


@dataclass
class MetricsRun:
    run_id: str
    is_synthetic: bool
    values: list[MetricValue] = field(default_factory=list)
    events: list[StructuralEvent] = field(default_factory=list)
    comparisons: list[PeriodComparison] = field(default_factory=list)
    attributions: list[Attribution] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    knowledge: list[KnowledgeRecord] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    custom_definitions: list[MetricDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
