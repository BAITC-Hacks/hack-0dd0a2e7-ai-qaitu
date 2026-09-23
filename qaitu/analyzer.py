from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from .models import AnalysisResult, Document, Finding, Fragment, Function, FunctionMatch, UnitChange


UNIT_WORDS = r"(?:департамент|отдел|управление|служба|сектор|центр|комитет|группа)"
UNIT_RE = re.compile(rf"\b({UNIT_WORDS}\s+[А-ЯЁA-Z][^.;:()\n]{{2,80}})", re.IGNORECASE)
FUNCTION_MARKERS = re.compile(
    r"\b(осуществляет|обеспечивает|проводит|контролирует|проверяет|разрабатывает|"
    r"согласовывает|утверждает|вед[её]т|организует|отвечает|формирует|оценивает|"
    r"подготавливает|выполняет|координирует|расследует|консультирует)\b",
    re.IGNORECASE,
)
AUDIT_MARKERS = {"аудит", "провер", "контрол", "оценк", "монитор"}
EXECUTION_MARKERS = {"выполн", "осуществл", "разработ", "веден", "ведение", "подготов", "реализ"}
STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "для", "от", "до", "из", "за", "при", "а", "или",
    "его", "ее", "их", "организации", "компании", "подразделение", "отдел", "департамент", "служба",
    "осуществляет", "обеспечивает", "проводит", "выполняет", "функции", "функцию",
}


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return " ".join(re.findall(r"[а-яa-z0-9]+", text))


def _tokens(text: str) -> set[str]:
    return {word for word in _norm(text).split() if len(word) > 2 and word not in STOPWORDS}


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    sequence = SequenceMatcher(None, _norm(a), _norm(b)).ratio()
    return 0.72 * jaccard + 0.28 * sequence


def _clean_unit(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;–—-")


def _extract_unit(fragment: Fragment) -> str | None:
    match = UNIT_RE.search(fragment.text)
    if not match:
        return None
    unit = _clean_unit(match.group(1))
    # Avoid swallowing common predicates when the source is one long sentence.
    unit = FUNCTION_MARKERS.split(unit, maxsplit=1)[0]
    return _clean_unit(unit)


def extract_units_and_functions(documents: list[Document]) -> tuple[dict[str, list[Fragment]], list[Function]]:
    units: dict[str, list[Fragment]] = defaultdict(list)
    functions: list[Function] = []
    current_unit: str | None = None
    fn_no = 0
    for document in documents:
        for fragment in document.fragments:
            explicit = _extract_unit(fragment)
            if explicit:
                current_unit = explicit
                units[current_unit].append(fragment)
            if current_unit and FUNCTION_MARKERS.search(fragment.text):
                chunks = re.split(r"\s*[;•]\s*|\n+", fragment.text)
                for chunk in chunks:
                    if FUNCTION_MARKERS.search(chunk) and len(_tokens(chunk)) >= 2:
                        fn_no += 1
                        functions.append(Function(f"{document.period}:fn:{fn_no}", current_unit, chunk.strip(), fragment))
    return dict(units), functions


def _best_match(query: str, candidates: list[str]) -> tuple[str | None, float]:
    if not candidates:
        return None, 0.0
    scored = sorted(((candidate, _similarity(query, candidate)) for candidate in candidates), key=lambda x: x[1], reverse=True)
    return scored[0]


def _match_units(before: dict[str, list[Fragment]], after: dict[str, list[Fragment]]) -> list[UnitChange]:
    result: list[UnitChange] = []
    unused_after = set(after)
    for old in before:
        new, score = _best_match(old, list(unused_after))
        if new is not None and score >= 0.48:
            old_type = _norm(old).split()[0]
            new_type = _norm(new).split()[0]
            status = "preserved" if score >= 0.78 and old_type == new_type else "transformed"
            result.append(UnitChange(status, old, new, round(score, 2), [before[old][0], after[new][0]]))
            unused_after.remove(new)
        else:
            result.append(UnitChange("removed", old, None, 1.0, [before[old][0]]))
    for new in sorted(unused_after):
        result.append(UnitChange("created", None, new, 1.0, [after[new][0]]))
    return result


def _match_functions(before: list[Function], after: list[Function]) -> list[FunctionMatch]:
    result: list[FunctionMatch] = []
    unused_after = set(range(len(after)))
    for old in before:
        if unused_after:
            idx, score = max(((idx, _similarity(old.text, after[idx].text)) for idx in unused_after), key=lambda x: x[1])
        else:
            idx, score = -1, 0.0
        if score >= 0.43:
            new = after[idx]
            same_unit = _similarity(old.unit, new.unit) >= 0.65
            status = "preserved" if same_unit and score >= 0.72 else ("moved" if not same_unit else "changed")
            result.append(FunctionMatch(old, new, round(score, 2), status))
            unused_after.remove(idx)
        else:
            result.append(FunctionMatch(old, None, round(score, 2), "lost"))
    for idx in sorted(unused_after):
        result.append(FunctionMatch(None, after[idx], 0.0, "new"))
    return result


def _find_duplicates(functions: list[Function]) -> list[Finding]:
    findings: list[Finding] = []
    for i, left in enumerate(functions):
        for right in functions[i + 1:]:
            if _similarity(left.unit, right.unit) >= 0.75:
                continue
            score = _similarity(left.text, right.text)
            if score >= 0.58:
                findings.append(Finding(
                    "duplicate",
                    f"Возможное дублирование: {left.unit} ↔ {right.unit}",
                    "Два разных подразделения имеют существенно похожие формулировки функций.",
                    round(score, 2), [left.source, right.source],
                    "Уточнить границы ответственности и назначить единственного владельца результата.",
                ))
    return findings


def _find_conflicts(functions: list[Function]) -> list[Finding]:
    grouped: dict[str, list[Function]] = defaultdict(list)
    for function in functions:
        grouped[function.unit].append(function)
    findings: list[Finding] = []
    for unit, items in grouped.items():
        audit = [f for f in items if re.search(r"\b(проверяет|контролирует|оценивает|проводит аудит)\b", f.text, re.I)]
        execution = [f for f in items if re.search(
            r"\b(осуществляет\s+(?!независимую оценку)|выполняет|вед[её]т\s+\S+\s+процедур)", f.text, re.I
        )]
        for check in audit:
            for execute in execution:
                overlap = len(_tokens(check.text) & _tokens(execute.text))
                if check.id != execute.id and overlap >= 2:
                    findings.append(Finding(
                        "conflict", f"Потенциальный конфликт ролей в «{unit}»",
                        "Одно подразделение участвует в выполнении процесса и в его контроле. Это индикатор для экспертной проверки, а не утверждение о нарушении.",
                        min(0.95, 0.55 + overlap * 0.08), [execute.source, check.source],
                        "Развести исполнение и независимую проверку либо документировать компенсирующий контроль.",
                    ))
                    break
            if findings and findings[-1].title.endswith(f"«{unit}»"):
                break
    return findings


def analyze_documents(before_docs: list[Document], after_docs: list[Document]) -> AnalysisResult:
    before_units, before_functions = extract_units_and_functions(before_docs)
    after_units, after_functions = extract_units_and_functions(after_docs)
    changes = _match_units(before_units, after_units)
    matches = _match_functions(before_functions, after_functions)
    findings: list[Finding] = []
    for match in matches:
        if match.status == "lost" and match.before:
            findings.append(Finding(
                "loss", f"Возможная потеря функции: {match.before.unit}",
                "В документах «после» не найдено достаточно близкой функции. Низкое текстовое сходство не исключает передачу функции под другой формулировкой.",
                round(1 - match.similarity, 2), [match.before.source],
                "Проверить, должна ли функция быть передана новому владельцу или исключена распорядительным документом.",
            ))
    findings.extend(_find_duplicates(after_functions))
    findings.extend(_find_conflicts(after_functions))
    warnings: list[str] = []
    if not before_units or not after_units:
        warnings.append("Не во всех комплектах распознаны подразделения. Проверьте качество текста и формулировки заголовков.")
    if not before_functions or not after_functions:
        warnings.append("Не во всех комплектах распознаны функции. Для сканированных PDF требуется предварительный OCR.")
    return AnalysisResult(changes, matches, findings, warnings)
