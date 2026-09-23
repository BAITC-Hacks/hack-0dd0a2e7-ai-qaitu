from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Fragment:
    id: str
    document: str
    text: str
    locator: str
    period: Literal["before", "after"]


@dataclass
class Document:
    name: str
    period: Literal["before", "after"]
    fragments: list[Fragment]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Function:
    id: str
    unit: str
    text: str
    source: Fragment
    norm_type: Literal["duty", "right", "prohibition"] = "duty"
    role: str = "execute"
    scope: str = ""
    context_sources: tuple[Fragment, ...] = ()
    owner_known: bool = True


@dataclass
class ConfidenceAssessment:
    score: float
    level: str
    priority: str
    method: str = "local-heuristic-v1"
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    metrics: dict[str, str | float | int | bool] = field(default_factory=dict)
    evidence: list[Fragment] = field(default_factory=list)


@dataclass
class Finding:
    kind: Literal["loss", "duplicate", "conflict"]
    title: str
    explanation: str
    confidence: float
    sources: list[Fragment] = field(default_factory=list)
    recommendation: str = ""
    matrix_row_id: str = ""
    assessment: ConfidenceAssessment | None = None


@dataclass
class UnitChange:
    status: Literal["preserved", "created", "removed", "transformed"]
    before: str | None
    after: str | None
    confidence: float
    sources: list[Fragment] = field(default_factory=list)


@dataclass
class FunctionMatch:
    before: Function | None
    after: Function | None
    similarity: float
    status: Literal["preserved", "moved", "changed", "lost", "new"]


@dataclass
class MatrixRow:
    id: str
    label: str
    before: list[Function]
    after: list[Function]
    status: str
    norm_type: str = "duty"
    notes: list[str] = field(default_factory=list)
    candidate_overlap: bool = False
    assessment: ConfidenceAssessment | None = None


@dataclass
class AnalysisResult:
    unit_changes: list[UnitChange]
    function_matches: list[FunctionMatch]
    findings: list[Finding]
    warnings: list[str] = field(default_factory=list)
    matrix_rows: list[MatrixRow] = field(default_factory=list)
    sources: list[Fragment] = field(default_factory=list)
    units_before: list[str] = field(default_factory=list)
    units_after: list[str] = field(default_factory=list)
    coverage: dict[str, int] = field(default_factory=dict)
    analysis_context: dict[str, str | bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
