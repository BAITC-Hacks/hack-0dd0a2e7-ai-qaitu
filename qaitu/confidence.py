"""Explainable heuristic scores, not calibrated probabilities of violations.

This module scores existing candidates; it never turns a low match into proof
of deletion or changes the analyzer's assignment/status decisions.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Callable

from .models import AnalysisResult, ConfidenceAssessment, Document, Function, MatrixRow


LEVELS = {
    "very_high": "🔴 Очень высокая",
    "high": "🟠 Высокая",
    "review": "🟡 Требует проверки",
    "weak": "⚪ Слабый сигнал",
}
PRIORITIES = {
    "very_high": "Высокий приоритет экспертной проверки",
    "high": "Высокий приоритет экспертной проверки",
    "review": "Обычный приоритет экспертной проверки",
    "weak": "Низкий приоритет экспертной проверки",
}
OBJECT_STOP = {"систем", "работ", "процесс", "план", "результат", "информац", "организ"}
METRIC_LABELS = {
    "owners_checked": "Проверено известных владельцев",
    "assignments_checked": "Проверено назначений после",
    "unknown_owners_after": "Назначений после без установленного владельца",
    "extraction_quality": "Качество извлечения",
    "package_completeness": "Полнота комплекта после",
    "norm_type": "Тип нормы", "owner_known_before": "Прежний владелец известен",
    "owner_known": "Все владельцы известны",
    "strongest_weighted_match": "Наибольшее взвешенное сходство",
    "nearest_text_similarity": "Ближайшее текстовое совпадение",
    "action_overlap": "Совпадение действий", "object_overlap": "Совпадение слов объекта",
    "nearest_owner": "Владелец ближайшего совпадения", "nearest_norm_type": "Тип нормы ближайшего совпадения",
    "possible_transfer": "Есть признаки возможного переноса",
    "possible_split": "Есть признаки возможного разделения",
    "match_coverage": "Взвешенное сходство сопоставленных назначений",
    "split_matched": "Составное сопоставление", "text_similarity": "Текстовое сходство",
    "scope_match": "Сравнение областей ответственности", "same_owner": "Общий владелец",
    "roles": "Роли", "reported_score": "Исходная самооценка модели",
    "raw_score": "Эвристический балл до ограничения", "score_cap": "Предел оценки с учётом ограничений",
    "distinct_sources": "Разные исходные пункты",
    "nearest_text_source": "Источник ближайшего текстового совпадения",
    "strongest_match_source": "Источник наибольшего взвешенного сходства",
}
PERCENT_METRICS = {"strongest_weighted_match", "nearest_text_similarity", "action_overlap", "object_overlap",
                   "match_coverage", "text_similarity", "reported_score", "raw_score", "score_cap"}


def metric_lines(value: ConfidenceAssessment) -> list[str]:
    lines = []
    for key, raw in value.metrics.items():
        if key in PERCENT_METRICS:
            display = f"{raw:.0%}"
        elif isinstance(raw, bool):
            display = "Да" if raw else "Нет"
        elif key in {"norm_type", "nearest_norm_type"}:
            display = {"duty": "Обязанность", "right": "Право", "prohibition": "Запрет"}.get(raw, raw)
        else:
            display = str(raw)
        lines.append(f"{METRIC_LABELS.get(key, key)}: {display}")
    return lines


def confidence_level(score: float) -> str:
    # Round once before assigning a band so 89.6% is never labelled 70–89%.
    percent = round(max(0.0, min(0.99, score)) * 100)
    return "very_high" if percent >= 90 else "high" if percent >= 70 else "review" if percent >= 40 else "weak"


def assessment(score: float, *, cap: float = .99, **details) -> ConfidenceAssessment:
    value = round(max(0.0, min(score, cap, .99)), 2)
    level = confidence_level(value)
    metrics = details.setdefault("metrics", {})
    metrics.update({"raw_score": round(score, 4), "score_cap": min(cap, .99)})
    return ConfidenceAssessment(value, level, PRIORITIES[level], **details)


def reported_assessment(score: float) -> ConfidenceAssessment:
    return assessment(
        score, cap=.89, method="llm-self-report",
        reasons=["Модель предложила кандидата с проверяемыми цитатами из загруженных документов."],
        limitations=["Оценка сообщена моделью; локальная разбивка по факторам отсутствует.",
                     "Полнота комплекта не проверена моделью. Это не вероятность нарушения и не подтверждение потери."],
        metrics={"reported_score": score, "package_completeness": "Не проверена моделью"},
    )


def confidence_counts(findings) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for finding in findings:
        level = finding.assessment.level if finding.assessment else confidence_level(finding.confidence)
        counts.setdefault(finding.kind, Counter())[level] += 1
    return counts


def assess_result(
    result: AnalysisResult, before_docs: list[Document], after_docs: list[Document], *,
    after_complete: bool,
    similarity: Callable[[str, str], float],
    tokens: Callable[[str], set[str]],
    actions: Callable[[str], set[str]],
) -> None:
    after = list({f.id: f for row in result.matrix_rows for f in row.after}.values())
    owners = {f.unit for f in after if f.owner_known}
    unknown = sum(not f.owner_known for f in after)
    known_ratio = 1 - unknown / len(after) if after else 0.0
    before_issues = [warning for doc in before_docs for warning in doc.warnings]
    after_issues = [warning for doc in after_docs for warning in doc.warnings]
    data_present = bool(before_docs and after_docs and after)
    clean = data_present and not (before_issues or after_issues)
    quality = "Без обнаруженных проблем извлечения" if clean else "Есть ограничения извлечения или недостаточно данных"
    result.analysis_context.update({
        "confidence_method": "local-heuristic-v1",
        "after_complete_user_declared": after_complete,
        "confidence_meaning": "Эвристическая уверенность в кандидате; не вероятность нарушения и не тяжесть последствий",
    })

    def signals(left: Function, right: Function) -> tuple[float, float, float]:
        la, ra = actions(left.text), actions(right.text)
        action = len(la & ra) / max(1, len(la | ra))
        lo = tokens(left.text) - la - OBJECT_STOP
        ro = tokens(right.text) - ra - OBJECT_STOP
        obj = len(lo & ro) / max(1, len(lo | ro))
        return similarity(left.text, right.text), action, obj

    def support(left: Function, right: Function) -> float:
        # Lexical overlap cannot make a prohibition preserve a right or duty.
        if left.norm_type != right.norm_type:
            return 0.0
        text, action, obj = signals(left, right)
        return .55 * text + .20 * action + .25 * obj

    def common_metrics() -> dict:
        return {
            "owners_checked": len(owners), "assignments_checked": len(after),
            "unknown_owners_after": unknown, "extraction_quality": quality,
            "package_completeness": "Заявлена пользователем" if after_complete else "Не подтверждена",
        }

    def limits() -> list[str]:
        notes = []
        if not after_complete:
            notes.append("Не подтверждено, что загружены все необходимые документы «после».")
        if not clean:
            notes.append("Ограничения извлечения снижают уверенность; проверьте предупреждения обработки.")
        if unknown:
            notes.append(f"У {unknown} назначений «после» владелец не установлен; эти тексты также участвовали в поиске.")
        return notes

    def loss_score(row: MatrixRow) -> ConfidenceAssessment:
        pairs = [(old, new) for old in row.before for new in after]
        nearest = max(pairs, key=lambda pair: similarity(pair[0].text, pair[1].text), default=None)
        compatible_pairs = [pair for pair in pairs if pair[0].norm_type == pair[1].norm_type]
        strongest_pair = max(compatible_pairs, key=lambda pair: support(*pair), default=None)
        strongest = support(*strongest_pair) if strongest_pair else 0.0
        old_known = bool(row.before) and all(f.owner_known for f in row.before)
        score = .45 + .38 * (1 - strongest) + .08 * old_known + .05 * known_ratio + .03 * bool(row.before)
        cap = .99 if after_complete and clean and not unknown else .89
        reasons = ["Старое назначение и его исходный пункт найдены.",
                   "Прямого соответствия среди извлечённых назначений «после» не найдено.",
                   f"Проверено известных владельцев: {len(owners)}; назначений: {len(after)}."]
        if old_known:
            reasons.append("Прежний владелец известен: " + ", ".join(sorted({f.unit for f in row.before})))
        limitations = limits() + ["Возможны сильная переформулировка или закрепление в другом документе."]
        metrics = common_metrics()
        metrics.update({"norm_type": row.norm_type, "owner_known_before": old_known,
                        "strongest_weighted_match": round(strongest, 4)})
        evidence = []
        transfer = any(old.unit != new.unit and new.owner_known and support(old, new) >= .55 for old, new in pairs)
        split = False
        for old in row.before:
            relevant = [new for new in after if new.norm_type == old.norm_type
                        and signals(old, new)[2] >= .35 and signals(old, new)[0] >= .30]
            old_actions = actions(old.text)
            covered = set().union(*(actions(new.text) & old_actions for new in relevant)) if relevant else set()
            split |= len({f.source.id for f in relevant}) >= 2 and len(covered) >= 2
        metrics.update({"possible_transfer": transfer, "possible_split": split})
        if nearest:
            text, action, obj = signals(*nearest)
            metrics.update({"nearest_text_similarity": round(text, 4), "action_overlap": round(action, 4),
                            "object_overlap": round(obj, 4), "nearest_owner": nearest[1].unit,
                            "nearest_norm_type": nearest[1].norm_type, "nearest_text_source": nearest[1].source.id})
            evidence = [nearest[1].source]
            if nearest[0].norm_type != nearest[1].norm_type:
                limitations.append("Ближайший текст содержит другой тип нормы и приведён только как контекст; он не подтверждает сохранение или перенос прежнего назначения.")
        if strongest_pair:
            metrics["strongest_match_source"] = strongest_pair[1].source.id
            if strongest_pair[1].source not in evidence:
                evidence.append(strongest_pair[1].source)
        if transfer or split or strongest >= .55:
            cap = min(cap, .69)
            limitations.append("Есть признаки возможного переноса, переформулировки или разделения; проверьте ближайшие назначения.")
        if not clean:
            cap = min(cap, .69)
        if not old_known or not data_present:
            cap = .39
        return assessment(score, cap=cap, reasons=reasons, limitations=limitations, metrics=metrics, evidence=evidence)

    for row in result.matrix_rows:
        if row.status == "lost" and row.norm_type != "prohibition":
            row.assessment = loss_score(row)
        elif row.status in {"preserved", "moved", "changed"} and row.before and row.after:
            # Bidirectional coverage avoids overconfidence based on one good pair.
            values = [max(support(old, new) for new in row.after) for old in row.before]
            values += [max(support(old, new) for old in row.before) for new in row.after]
            coverage = sum(values) / len(values)
            known = all(f.owner_known for f in row.before + row.after)
            same_norm = len({f.norm_type for f in row.before + row.after}) == 1
            score = .10 + .65 * coverage + .10 * known + .10 * same_norm + .04 * clean
            split = any("Составная прежняя" in note for note in row.notes)
            cap = .89 if split else .99
            if not clean:
                cap = min(cap, .69)
            reasons = ["Найдены назначения и источники обеих редакций.",
                       "Набор владельцев изменился." if row.status == "moved" else "Набор владельцев сохранён."]
            metrics = common_metrics()
            metrics.update({"match_coverage": round(coverage, 4), "norm_type": row.norm_type,
                            "owner_known": known, "split_matched": split})
            row.assessment = assessment(score, cap=cap, reasons=reasons,
                limitations=limits() + (["Составное сопоставление требует проверки полноты всех частей."] if split else []),
                metrics=metrics)
            row.assessment.priority = "Проверка сопоставления по источникам"

    row_map = {row.id: row for row in result.matrix_rows}
    for finding in result.findings:
        row = row_map.get(finding.matrix_row_id)
        if finding.kind == "loss" and row and row.assessment:
            finding.assessment = row.assessment
        elif finding.kind == "duplicate" and row:
            pairs = [(a, b) for a, b in combinations(row.after, 2) if a.unit != b.unit and a.owner_known and b.owner_known]
            pair = max(pairs, key=lambda pair: support(*pair), default=None)
            if pair:
                text, action, obj = signals(*pair)
                scoped = bool(pair[0].scope and pair[1].scope)
                same_scope = scoped and pair[0].scope == pair[1].scope
                distinct = pair[0].source.id != pair[1].source.id
                score = .15 + .35 * text + .20 * action + .20 * obj + .05 + .05 * same_scope
                cap = .99 if same_scope else .89 if not scoped else .69
                if not distinct:
                    cap = min(cap, .69)
                finding.assessment = assessment(score, cap=cap if clean else min(cap, .69),
                    reasons=["Обязанность закреплена за разными известными владельцами.", "Проверены сходство формулировок, действий и объектов."],
                    limitations=limits() + ([] if same_scope else ["Совпадение областей ответственности не подтверждено; совместное выполнение может быть обоснованным."])
                        + ([] if distinct else ["Владельцы указаны в общем пункте: это не два независимых подтверждения дублирования."]),
                    metrics={**common_metrics(), "text_similarity": round(text, 4), "action_overlap": round(action, 4),
                             "object_overlap": round(obj, 4), "distinct_sources": distinct,
                             "scope_match": "Совпадает" if same_scope else "Различается" if scoped else "Не установлено"},
                    evidence=list({f.source.id: f.source for f in pair}.values()))
        elif finding.kind == "conflict":
            source_ids = {source.id for source in finding.sources}
            function_ids = set(finding.function_ids)
            items = [f for f in after if f.source.id in source_ids and f.norm_type == "duty"
                     and (not function_ids or f.id in function_ids)]
            pairs = [(a, b) for a in items for b in items if a.id != b.id and a.role == "execute" and b.role == "control"
                     and a.unit == b.unit and a.owner_known and b.owner_known
                     and not (a.scope and b.scope and a.scope != b.scope)]
            binding_issue = None
            if function_ids:
                if len(function_ids) != 2 or {f.id for f in items} != function_ids:
                    pairs = []
                if not pairs:
                    binding_issue = "Исходные назначения конфликта не удалось однозначно связать с владельцем, ролями и приведёнными источниками."
            elif len({a.unit for a, _ in pairs}) > 1:
                pairs = []
                binding_issue = "Общие исходные пункты относятся к нескольким владельцам; без ID исходных назначений владелец этого кандидата не установлен."
            pair = max(pairs, key=lambda pair: signals(*pair)[2], default=None)
            if not pair:
                finding.assessment = assessment(min(finding.confidence, .39),
                    limitations=limits() + [binding_issue or "Недостаточно извлечённых признаков для связи исполнения и контроля с одним владельцем."])
            if pair:
                text, action, obj = signals(*pair)
                same_scope = bool(pair[0].scope and pair[0].scope == pair[1].scope)
                finding.assessment = assessment(.20 + .50 * obj + .15 + .10 * clean + .05 * same_scope,
                    cap=.99 if clean and same_scope else .89 if clean else .69,
                    reasons=["Исполнение и контроль закреплены за одним известным владельцем.", "В текстах найден общий объект деятельности."],
                    limitations=limits() + ["Нужно подтвердить независимость контроля и совпадение области; сходство текста не устанавливает нарушение."],
                    metrics={**common_metrics(), "object_overlap": round(obj, 4), "same_owner": pair[0].unit,
                             "roles": "Исполнение и контроль", "scope_match": "Совпадает" if same_scope else "Не установлено"},
                    evidence=[f.source for f in pair])
        if finding.assessment is None:
            finding.assessment = assessment(min(finding.confidence, .39),
                limitations=["Недостаточно извлечённых признаков для объяснимой оценки."])
        finding.confidence = finding.assessment.score
