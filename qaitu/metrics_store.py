"""Atomic stage snapshots and a deduplicated, namespace-separated case registry."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

from .metrics_models import KnowledgeRecord, MetricsRun

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def _root(storage_root):
    return Path(storage_root) if storage_root is not None else DEFAULT_ROOT


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".metrics-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _run_dir(run, storage_root):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run.run_id):
        raise ValueError("Некорректный идентификатор запуска")
    return _root(storage_root) / "runs" / run.run_id / "metrics"


def persist_run(run: MetricsRun, storage_root=None):
    directory = _run_dir(run, storage_root)
    for stage, field in (("M1_ingest", "values"), ("M2_events", "events"),
                         ("M3_alignment", "comparisons"), ("M4_compare", "comparisons"),
                         ("M5_attribute", "attributions"), ("M6_hypotheses", "hypotheses"),
                         ("M7_knowledge", "knowledge"), ("M8_recommend", "recommendations")):
        atomic_json(directory / f"{stage}.json", {"is_synthetic": run.is_synthetic,
                    "records": [asdict(value) for value in getattr(run, field)]})
    atomic_json(directory / "M9_report.json", run.to_dict())


@contextmanager
def _connection(storage_root):
    path = _root(storage_root) / "data" / "kb.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Set permissions before connecting; contents may refer to private reports.
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS cases (namespace TEXT, case_id TEXT, pattern_id TEXT, payload TEXT, PRIMARY KEY(namespace,case_id))")
        connection.execute("CREATE TABLE IF NOT EXISTS decisions (namespace TEXT, recommendation_id TEXT, decision TEXT, PRIMARY KEY(namespace,recommendation_id))")
        with connection:
            yield connection
    finally:
        connection.close()


def _namespace(synthetic):
    return "synthetic" if synthetic else "real"


def update_knowledge(run: MetricsRun, storage_root=None):
    events = {event.id: event for event in run.events}
    comparisons = {comparison.id: comparison for comparison in run.comparisons}
    with _connection(storage_root) as connection:
        namespace = _namespace(run.is_synthetic)
        # Re-evaluate every submitted event, even if its new settings produce no
        # comparisons. Independent events sharing a date/scope cannot revoke it.
        reviewed = {_digest(event.id, event.effective_date) for event in run.events}
        for case_id, encoded in connection.execute("SELECT case_id,payload FROM cases WHERE namespace=?", (namespace,)).fetchall():
            prior = json.loads(encoded)
            identity = prior.get("event_identity")
            if identity is None or identity in reviewed:
                connection.execute("DELETE FROM cases WHERE namespace=? AND case_id=?", (namespace, case_id))
        for attribution in run.attributions:
            event = events.get(attribution.event_id)
            comparison = comparisons.get(attribution.comparison_id)
            if event is None or comparison is None:
                continue
            pattern = f"{event.type}: {' '.join(event.mechanism.casefold().split())}"
            pattern_id = _digest(pattern, comparison.metric)
            # A longer time series or a different filename is still the same event.
            case_id = _digest(event.id, event.effective_date, comparison.unit_scope, comparison.metric)
            namespace = _namespace(run.is_synthetic)
            if attribution.confidence not in {"medium", "high"}:
                connection.execute("DELETE FROM cases WHERE namespace=? AND case_id=?", (namespace, case_id))
                continue
            payload = {"pattern": pattern, "event_date": event.effective_date,
                       "event_identity": _digest(event.id, event.effective_date), "event_id": event.id,
                       "unit": comparison.unit_scope, "metric": comparison.metric,
                       "confidence": attribution.confidence, "attribution_id": attribution.id,
                       "run_id": run.run_id, "delta_pct": comparison.delta_pct,
                       "delta_abs": comparison.delta_abs, "adjusted_delta": comparison.adjusted_delta,
                       "method": comparison.method, "evidence": attribution.evidence,
                       "source_values": [asdict(v) for v in run.values if v.id in comparison.metric_ids]}
            connection.execute("INSERT INTO cases VALUES (?,?,?,?) ON CONFLICT(namespace,case_id) DO UPDATE SET payload=excluded.payload, pattern_id=excluded.pattern_id",
                               (namespace, case_id, pattern_id, json.dumps(payload, ensure_ascii=False, allow_nan=False)))
    return load_knowledge(run.is_synthetic, storage_root)


def load_knowledge(is_synthetic=False, storage_root=None):
    with _connection(storage_root) as connection:
        rows = connection.execute("SELECT case_id, pattern_id, payload FROM cases WHERE namespace=? ORDER BY pattern_id,case_id", (_namespace(is_synthetic),)).fetchall()
    groups = {}
    for case_id, pattern_id, encoded in rows:
        groups.setdefault(pattern_id, []).append((case_id, json.loads(encoded)))
    records = []
    for pattern_id, cases in groups.items():
        items = [entry for _, entry in cases]
        diversity = len({(item["event_date"], item["unit"]) for item in items})
        deltas = [item["delta_pct"] for item in items if item["delta_pct"] is not None]
        numeric_allowed = diversity >= 3 and diversity == len(cases) and len(deltas) == len(cases)
        mean = sum(deltas) / len(deltas) if numeric_allowed else None
        signs = {"up" if item["delta_abs"] > 0 else "down" if item["delta_abs"] < 0 else "flat" for item in items}
        records.append(KnowledgeRecord(
            id=f"kb_{pattern_id}", change_pattern=items[0]["pattern"],
            context={"units": sorted({item["unit"] for item in items}), "dates": sorted({item["event_date"] for item in items}),
                     "cases": items, "interpretation": "единичное наблюдение" if diversity == 1 else "наблюдалось в похожих случаях"},
            observed_effect={"metric": items[0]["metric"], "direction": next(iter(signs)) if len(signs) == 1 else "mixed",
                             "delta_pct": mean, "quantitative_allowed": numeric_allowed,
                             "note": "Среднее наблюдение; не обещание будущего эффекта"},
            confidence="high" if all(item["confidence"] == "high" for item in items) else "medium",
            cases=[case_id for case_id, _ in cases], n_cases=diversity, is_synthetic=is_synthetic, diversity=diversity))
    return records


def save_decision(run: MetricsRun, recommendation_id: str, decision: str, storage_root=None):
    if decision not in {"accept", "reject", "discuss"}:
        raise ValueError("Неизвестное решение")
    recommendation = next((r for r in run.recommendations if r.id == recommendation_id), None)
    if recommendation is None:
        raise ValueError("Рекомендация не найдена")
    with _connection(storage_root) as connection:
        connection.execute("INSERT INTO decisions VALUES (?,?,?) ON CONFLICT(namespace,recommendation_id) DO UPDATE SET decision=excluded.decision",
                           (_namespace(run.is_synthetic), recommendation_id, decision))
    recommendation.decision = decision
    persist_run(run, storage_root)


def restore_decisions(run: MetricsRun, storage_root=None):
    with _connection(storage_root) as connection:
        decisions = dict(connection.execute("SELECT recommendation_id,decision FROM decisions WHERE namespace=?", (_namespace(run.is_synthetic),)))
    for recommendation in run.recommendations:
        recommendation.decision = decisions.get(recommendation.id, "discuss")


def _mechanism_key(event):
    return _digest(event.type, sorted(event.units), event.evidence)


def cache_mechanisms(events, storage_root=None):
    with _connection(storage_root) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS mechanisms (namespace TEXT, source_key TEXT, payload TEXT, PRIMARY KEY(namespace,source_key))")
        for event in events:
            key = (_namespace(event.is_synthetic), _mechanism_key(event))
            if event.mechanism_confirmed:
                payload = json.dumps({"metric_codes": event.metric_codes, "mechanism": event.mechanism}, ensure_ascii=False)
                connection.execute("INSERT INTO mechanisms VALUES (?,?,?) ON CONFLICT(namespace,source_key) DO UPDATE SET payload=excluded.payload", (*key, payload))
            else:
                connection.execute("DELETE FROM mechanisms WHERE namespace=? AND source_key=?", key)


def apply_cached_mechanisms(events, storage_root=None):
    with _connection(storage_root) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS mechanisms (namespace TEXT, source_key TEXT, payload TEXT, PRIMARY KEY(namespace,source_key))")
        for event in events:
            row = connection.execute("SELECT payload FROM mechanisms WHERE namespace=? AND source_key=?", (_namespace(event.is_synthetic), _mechanism_key(event))).fetchone()
            if row:
                payload = json.loads(row[0])
                event.mechanism, event.metric_codes = payload["mechanism"], payload["metric_codes"]
                event.mechanism_confirmed = True
    return events
