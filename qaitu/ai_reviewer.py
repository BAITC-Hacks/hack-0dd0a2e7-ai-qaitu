from __future__ import annotations

import json

from .models import AnalysisResult, Document, Finding


SYSTEM_PROMPT = """Ты — агент внутреннего контроля, второй этап гибридного анализа реорганизации.
Работай ТОЛЬКО с переданным каталогом источников. Не используй внешние знания.
Найди существенные потери функций, дублирование и конфликт ролей, которые мог пропустить
лексический алгоритм. Конфликт ролей — когда одно подразделение исполняет процесс и независимо
проверяет тот же процесс. Каждый вывод обязан содержать source_ids из каталога. Если доказательств
недостаточно, не создавай вывод. Формулируй вероятностно: «возможный», «потенциальный».
Ответь только JSON-объектом: {"findings": [{"kind": "loss|duplicate|conflict",
"title": "...", "explanation": "...", "confidence": 0.0, "source_ids": ["..."],
"recommendation": "..."}]}.
"""


def review_with_llm(
    before_docs: list[Document],
    after_docs: list[Document],
    base_result: AnalysisResult,
    api_key: str,
    model: str = "gpt-4.1-mini",
) -> list[Finding]:
    """Run a grounded second-pass review and discard any untraceable model output."""
    from openai import OpenAI

    fragments = [f for doc in before_docs + after_docs for f in doc.fragments]
    source_map = {fragment.id: fragment for fragment in fragments}
    catalog = [{
        "source_id": f.id, "period": f.period, "document": f.document,
        "locator": f.locator, "text": f.text[:1500],
    } for f in fragments]
    payload = {
        "sources": catalog,
        "algorithm_findings": [{
            "kind": item.kind, "title": item.title,
            "source_ids": [source.id for source in item.sources],
        } for item in base_result.findings],
    }
    # Bound the material sent to a third-party API and make truncation explicit.
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) > 90_000:
        raise ValueError("Комплект слишком велик для LLM-проверки прототипа (лимит 90 000 символов).")

    response = OpenAI(api_key=api_key).responses.create(
        model=model,
        store=False,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": encoded},
        ],
        text={"format": {"type": "json_object"}},
    )
    if response.status != "completed":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
        raise RuntimeError(f"LLM-проверка не завершена: {reason}")
    message = next((item for item in response.output if item.type == "message"), None)
    content = message.content[0] if message and message.content else None
    if content and content.type == "refusal":
        raise RuntimeError(f"LLM отказался от обработки: {content.refusal}")
    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    data = json.loads(raw)
    findings: list[Finding] = []
    allowed_kinds = {"loss", "duplicate", "conflict"}
    for item in data.get("findings", []):
        source_ids = item.get("source_ids", [])
        # Fail closed: hallucinated or absent citations never reach the report.
        if item.get("kind") not in allowed_kinds or not source_ids:
            continue
        if any(source_id not in source_map for source_id in source_ids):
            continue
        findings.append(Finding(
            kind=item["kind"],
            title=str(item.get("title", "Вывод LLM-проверки")),
            explanation=str(item.get("explanation", "")),
            confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
            sources=[source_map[source_id] for source_id in source_ids],
            recommendation=str(item.get("recommendation", "Проверить вывод ответственным сотрудником.")),
        ))
    return findings
