"""Bounded optional LLM steps. Numeric calculations and confidence remain in code."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re

from .config import load_openai_settings

PROMPTS = Path(__file__).with_name("llm") / "prompts" / "metrics"
CAUSAL = re.compile(r"привел[оаи]?\s+к|вызвал[оаи]?|благодаря", re.I)
MAX_CHARS = 150_000
SAFE_ERROR = "ИИ-этап метрик не завершён; сохранены локальные расчёты и очередь проверки."


def _obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


S = {"type": "string"}
STRINGS = _array(S)


def _request(name, payload, schema, *, client=None, model=None):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if len(encoded) > MAX_CHARS:
        raise ValueError(SAFE_ERROR)
    try:
        if client is None:
            from openai import OpenAI
            settings = load_openai_settings()
            if not settings.api_key:
                raise ValueError(SAFE_ERROR)
            client = OpenAI(api_key=settings.api_key, max_retries=0)
            model = model or settings.model
        response = client.responses.create(
            model=model or "gpt-5.4-mini", store=False, timeout=45, max_output_tokens=6500,
            input=[{"role": "system", "content": (PROMPTS / f"{name}.md").read_text()},
                   {"role": "user", "content": encoded}],
            text={"format": {"type": "json_schema", "name": f"metrics_{name}", "strict": True, "schema": schema}},
            **({"reasoning": {"effort": "low"}} if (model or "gpt-5.4-mini").startswith("gpt-5.4") else {}))
        if response.status != "completed" or not isinstance(response.output_text, str) or len(response.output_text) > 80_000:
            raise ValueError(SAFE_ERROR)
        def invalid(_):
            raise ValueError(SAFE_ERROR)
        return json.loads(response.output_text, parse_constant=invalid)
    except Exception:
        # Never surface provider exceptions containing report text or credentials.
        raise ValueError(SAFE_ERROR) from None


def extract_metrics(fragments, catalog, unit_scope=None, is_synthetic=False, *, client=None, model=None):
    record = {"metric": S, "value": {"type": "number"}, "unit_scope": S, "period_start": S,
              "period_end": S, "granularity": {"type": "string", "enum": ["month", "quarter", "year"]},
              "source_id": S, "quote": S, "unit": S}
    schema = _obj({"values": _array(_obj(record)),
                   "ambiguous": _array(_obj({"source_id": S, "quote": S, "reason": S})),
                   "unknown_metrics": _array(_obj({**{k: v for k, v in record.items() if k != "metric"}, "name": S}))})
    return _request("extract_metrics", {"fragments": fragments, "catalog": catalog,
                    "unit_scope": unit_scope, "is_synthetic": is_synthetic}, schema, client=client, model=model)


def _text_ok(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 3000 and not CAUSAL.search(value)


def _numbers_grounded(text, facts):
    # Allow ordinary display rounding, but reject a number absent from the input.
    pattern = r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
    known = [float(token.replace(",", ".")) for token in re.findall(pattern, json.dumps(facts, ensure_ascii=False))]
    for token in re.findall(pattern, text):
        value = float(token.replace(",", "."))
        if not any(math.isclose(value, number, rel_tol=0.0005, abs_tol=0.005) for number in known):
            return False
    return True


def _index(records, field, expected):
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(SAFE_ERROR)
    index = {}
    for record in records:
        if not isinstance(record, dict) or record.get(field) not in expected or record[field] in index:
            raise ValueError(SAFE_ERROR)
        index[record[field]] = record
    return index


def enhance_attributions(values, events, comparisons, attributions, *, client=None, model=None):
    """A rejected or invalid pair never gets promoted by the AI layer."""
    if not attributions:
        return attributions, []
    selected = attributions[:8]
    ids = {a.id for a in selected}
    related_events = [e for e in events if any(e.id == a.event_id for a in selected)]
    comparison_ids = {a.comparison_id for a in selected}
    related_comparisons = [c for c in comparisons if c.id in comparison_ids]
    metric_ids = {i for c in related_comparisons for i in c.metric_ids + c.normalization_sources}
    metric_ids.update(i for a in selected for i in a.evidence.get("metrics", []))
    payload = {"attributions": [asdict(a) for a in selected], "comparisons": [asdict(c) for c in related_comparisons],
               "events": [asdict(e) for e in events], "values": [asdict(v) for v in values if v.id in metric_ids]}
    warnings = []
    if len(attributions) > len(selected):
        warnings.append("ИИ рассмотрел первые 8 связей; остальные проверены локальными правилами.")
    try:
        schema = _obj({"attributions": _array(_obj({"attribution_id": S, "claim": S, "mechanism": S,
                      "alternative_explanations": STRINGS, "metric_ids": STRINGS, "clause_ids": STRINGS}))})
        data = _request("attribute", payload, schema, client=client, model=model)
        drafts = _index(data["attributions"], "attribution_id", ids)
        original = {a.id: a for a in selected}
        for aid, draft in drafts.items():
            allowed_metrics = set(original[aid].evidence.get("metrics", []))
            allowed_clauses = {str(d.get("clause_id") or d.get("source_id") or d.get("id"))
                               for event in related_events if event.id == original[aid].event_id for d in event.evidence}
            numeric_context = {"original": asdict(original[aid]), "comparison": next(asdict(c) for c in related_comparisons if c.id == original[aid].comparison_id)}
            if not _numbers_grounded(draft["claim"], numeric_context):
                raise ValueError(SAFE_ERROR)
            if not all(_text_ok(draft[k]) for k in ("claim", "mechanism")) or not draft["alternative_explanations"]:
                raise ValueError(SAFE_ERROR)
            if not _numbers_grounded(draft["mechanism"], payload) or not all(_text_ok(x) and _numbers_grounded(x, payload) for x in draft["alternative_explanations"]):
                raise ValueError(SAFE_ERROR)
            if not draft["metric_ids"] or not set(draft["metric_ids"]) <= allowed_metrics or not draft["clause_ids"] or not set(draft["clause_ids"]) <= allowed_clauses:
                raise ValueError(SAFE_ERROR)
        review_schema = _obj({"reviews": _array(_obj({"attribution_id": S,
                             "outcome": {"type": "string", "enum": ["upheld", "weakened", "refuted"]},
                             "reasoning": S, "counter_evidence": STRINGS}))})
        reviews = _index(_request("challenge_attribution", {**payload, "drafts": list(drafts.values())}, review_schema,
                         client=client, model=model)["reviews"], "attribution_id", ids)
        from .metrics_analysis import calculate_confidence
        updated = []
        event_index = {e.id: e for e in events}
        comparison_index = {c.id: c for c in comparisons}
        ranking = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}
        for a in attributions:
            if a.id not in ids:
                updated.append(a)
                continue
            review, draft = reviews[a.id], drafts[a.id]
            if not _text_ok(review["reasoning"]) or not _numbers_grounded(review["reasoning"], payload) or review["outcome"] not in {"upheld", "weakened", "refuted"} or not all(_text_ok(s) and _numbers_grounded(s, payload) for s in review["counter_evidence"]):
                raise ValueError(SAFE_ERROR)
            # Code-derived facts and confirmed mechanism stay visible. AI contributes
            # a reviewed explanation but cannot increase the rule-based confidence.
            outcome = min((a.opponent_outcome, review["outcome"]), key={"refuted": 0, "weakened": 1, "upheld": 2}.get)
            confidence, reasons = calculate_confidence(event_index[a.event_id], comparison_index[a.comparison_id], outcome,
                                                       alternative_events=a.alternative_explanations)
            confidence = min((confidence, a.confidence), key=ranking.get)
            updated.append(replace(a, claim=draft["claim"] if outcome != "refuted" else a.claim,
                                   confidence=confidence, confidence_reasons=list(dict.fromkeys(a.confidence_reasons + reasons)),
                                   alternative_explanations=list(dict.fromkeys(a.alternative_explanations + draft["alternative_explanations"] + review["counter_evidence"])),
                                   opponent_outcome=outcome, opponent_reasoning=review["reasoning"], method="llm+rules"))
        return updated, warnings
    except Exception:
        return attributions, warnings + [SAFE_ERROR]


def refine_recommendations(recommendations, attributions, knowledge, *, client=None, model=None):
    if not recommendations:
        return recommendations, []
    selected = recommendations[:8]
    try:
        payload = {"recommendations": [asdict(r) for r in selected], "attributions": [asdict(a) for a in attributions], "knowledge": [asdict(k) for k in knowledge]}
        data = _request("recommend", payload,
                        _obj({"recommendations": _array(_obj({"recommendation_id": S, "action": S, "risks": STRINGS}))}),
                        client=client, model=model)
        index = _index(data["recommendations"], "recommendation_id", {r.id for r in selected})
        updated = []
        for recommendation in recommendations:
            if recommendation.id not in index:
                updated.append(recommendation)
                continue
            item = index[recommendation.id]
            if not _text_ok(item["action"]) or re.search(r"\d|%", item["action"]) or not all(_text_ok(r) and _numbers_grounded(r, payload) for r in item["risks"]):
                raise ValueError(SAFE_ERROR)
            # Free-form model actions are never published: even an advisory note
            # could assign a duty to an unsupported department. Keep rule actions.
            questions = ["Вопрос ИИ для проверки человеком (не установленный факт): " + risk for risk in item["risks"]]
            updated.append(replace(recommendation, risks=list(dict.fromkeys(recommendation.risks + questions))))
        return updated, []
    except Exception:
        return recommendations, [SAFE_ERROR]


def propose_mechanisms(events, catalog, *, client=None, model=None):
    """Unconfirmed suggestions only; source quotes and metric codes are checked."""
    if not events:
        return [], []
    selected = events[:12]
    try:
        payload = {"task": "Предложи связи функция→метрика для подтверждения человеком. Для каждого события укажи дословную цитату, metric_codes и осторожное объяснение. Не подтверждай механизм самостоятельно.",
                   "events": [asdict(event) for event in selected], "catalog": catalog}
        schema = _obj({"suggestions": _array(_obj({"event_id": S, "metric_codes": STRINGS, "mechanism": S, "quote": S}))})
        data = _request("attribute", payload, schema, client=client, model=model)
        suggestions = data["suggestions"]
        by_id = {event.id: event for event in selected}
        codes = {entry["code"] for entry in catalog}
        result, seen = [], set()
        for suggestion in suggestions:
            event = by_id.get(suggestion.get("event_id"))
            if event is None or event.id in seen or not _text_ok(suggestion["mechanism"]):
                raise ValueError(SAFE_ERROR)
            seen.add(event.id)
            quote = suggestion["quote"]
            if not isinstance(quote, str) or len(quote.strip()) < 12 or not any(quote in evidence.get("quote", "") for evidence in event.evidence):
                raise ValueError(SAFE_ERROR)
            if not suggestion["metric_codes"] or not set(suggestion["metric_codes"]) <= codes:
                raise ValueError(SAFE_ERROR)
            result.append(replace(event, metric_codes=suggestion["metric_codes"], mechanism=suggestion["mechanism"], mechanism_confirmed=False))
        replacements = {event.id: event for event in result}
        return [replacements.get(event.id, event) for event in events], []
    except Exception:
        return list(events), [SAFE_ERROR]
