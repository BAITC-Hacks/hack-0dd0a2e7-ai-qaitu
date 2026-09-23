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


@dataclass(frozen=True)
class Function:
    id: str
    unit: str
    text: str
    source: Fragment


@dataclass
class Finding:
    kind: Literal["loss", "duplicate", "conflict"]
    title: str
    explanation: str
    confidence: float
    sources: list[Fragment] = field(default_factory=list)
    recommendation: str = ""


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
class AnalysisResult:
    unit_changes: list[UnitChange]
    function_matches: list[FunctionMatch]
    findings: list[Finding]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
