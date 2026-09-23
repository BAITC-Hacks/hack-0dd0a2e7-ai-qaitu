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
SECTION_RE = re.compile(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?\.\s*(.*)")
ORG_ITEM_RE = re.compile(r"^\s*[а-яa-z]\.?\s+", re.I)
AUDIT_MARKERS = {"аудит", "провер", "контрол", "оценк", "монитор"}
EXECUTION_MARKERS = {"выполн", "осуществл", "разработ", "веден", "ведение", "подготов", "реализ"}
STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "для", "от", "до", "из", "за", "при", "а", "или",
    "его", "ее", "их", "организации", "компании", "подразделение", "отдел", "департамент", "служба",
    "осуществляет", "обеспечивает", "проводит", "выполняет", "функции", "функцию",
}


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"проектов документации,? регламентирующей работу", "внд", text)
    text = re.sub(r"внутренних нормативных документов", "внд", text)
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


def _structured_owner(section: str, heading: str, units: dict[str, list[Fragment]]) -> str:
    if section == "3" and "ДИТААД" in heading and "ДОА" in heading:
        return "ДИТААД и ДОА (совместные обязанности)"
    if section == "3":
        return "Директор направления внутреннего аудита"
    normalized = heading.lower().replace("департамента", "департамент")
    for unit in units:
        if _similarity(unit, normalized) >= 0.55:
            return unit
    return heading.strip(" :")


def _extract_structured_document(document: Document) -> tuple[dict[str, list[Fragment]], list[Function]] | None:
    """Parse provisions with an explicit 3.4 structure and numbered 2.4/5.x duties."""
    if not any(re.match(r"^\s*3\.4\.\s+.*(?:состоит|структурн)", f.text, re.I) for f in document.fragments):
        return None

    units: dict[str, list[Fragment]] = defaultdict(list)
    in_structure = False
    for fragment in document.fragments:
        match = SECTION_RE.match(fragment.text)
        if match and match.group(1) == "3" and match.group(2) == "4" and match.group(3) is None:
            in_structure = True
            continue
        if in_structure and match and match.group(1) == "3" and match.group(2) != "4":
            in_structure = False
        if in_structure and ORG_ITEM_RE.match(fragment.text):
            unit = _extract_unit(fragment)
            if unit:
                units[unit].append(fragment)

    functions: list[Function] = []
    owners: dict[str, str] = {}
    for fragment in document.fragments:
        match = SECTION_RE.match(fragment.text)
        if not match:
            continue
        major, section, item, body = match.groups()
        if major == "5" and item is None and section in {"3", "4", "5"}:
            owners[section] = _structured_owner(section, body, units)
        elif major == "5" and item and section in owners and section in {"3", "4", "5"}:
            if len(_tokens(body)) >= 3 and not body.strip().startswith(";"):
                functions.append(Function(
                    f"{document.period}:fn:{len(functions) + 1}", owners[section], body.strip(), fragment,
                ))
        elif major == "2" and section == "4" and item and len(_tokens(body)) >= 3:
            functions.append(Function(
                f"{document.period}:fn:{len(functions) + 1}", "БВА (общие функции)", body.strip(), fragment,
            ))
    return dict(units), functions


def extract_units_and_functions(documents: list[Document]) -> tuple[dict[str, list[Fragment]], list[Function]]:
    units: dict[str, list[Fragment]] = defaultdict(list)
    functions: list[Function] = []
    current_unit: str | None = None
    fn_no = 0
    for document in documents:
        structured = _extract_structured_document(document)
        if structured is not None:
            document_units, document_functions = structured
            for unit, sources in document_units.items():
                units[unit].extend(sources)
            for function in document_functions:
                fn_no += 1
                functions.append(Function(f"{document.period}:fn:{fn_no}", function.unit, function.text, function.source))
            continue
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


def _function_body(function: Function) -> str:
    marker = FUNCTION_MARKERS.search(function.text)
    return function.text[marker.start():] if marker else function.text


def _function_profile_similarity(left: list[Function], right: list[Function]) -> float:
    if not left or not right:
        return 0.0
    left_texts = [_function_body(function) for function in left]
    right_texts = [_function_body(function) for function in right]
    left_coverage = sum(max(_similarity(text, other) for other in right_texts) for text in left_texts) / len(left_texts)
    right_coverage = sum(max(_similarity(text, other) for other in left_texts) for text in right_texts) / len(right_texts)
    return (left_coverage + right_coverage) / 2


def _match_units(
    before: dict[str, list[Fragment]], after: dict[str, list[Fragment]],
    before_functions: list[Function], after_functions: list[Function],
) -> list[UnitChange]:
    result: list[UnitChange] = []
    old_profiles: dict[str, list[Function]] = defaultdict(list)
    new_profiles: dict[str, list[Function]] = defaultdict(list)
    for function in before_functions:
        old_profiles[function.unit].append(function)
    for function in after_functions:
        new_profiles[function.unit].append(function)

    candidates: list[tuple[float, float, float, str, str]] = []
    for old in before:
        for new in after:
            name_score = _similarity(old, new)
            function_score = _function_profile_similarity(old_profiles[old], new_profiles[new])
            score = 0.4 * name_score + 0.6 * function_score if function_score else name_score
            # A complete rename needs several corroborating functions, not one generic sentence.
            enough_evidence = name_score >= 0.25 or (
                min(len(old_profiles[old]), len(new_profiles[new])) >= 2 and function_score >= 0.75
            )
            if score >= 0.48 and enough_evidence:
                candidates.append((score, name_score, function_score, old, new))

    used_before: set[str] = set()
    used_after: set[str] = set()
    for score, name_score, _, old, new in sorted(candidates, key=lambda item: (-item[0], item[3], item[4])):
        if old in used_before or new in used_after:
            continue
        same_type = _norm(old).split()[0] == _norm(new).split()[0]
        status = "preserved" if name_score >= 0.78 and same_type else "transformed"
        result.append(UnitChange(status, old, new, round(score, 2), [before[old][0], after[new][0]]))
        used_before.add(old)
        used_after.add(new)
    for old in before:
        if old not in used_before:
            result.append(UnitChange("removed", old, None, 1.0, [before[old][0]]))
    for new in sorted(set(after) - used_after):
        result.append(UnitChange("created", None, new, 1.0, [after[new][0]]))
    return result


def _match_functions(before: list[Function], after: list[Function]) -> list[FunctionMatch]:
    result: list[FunctionMatch] = []
    matched_after: set[int] = set()
    for old in before:
        if after:
            idx, score = max(((idx, _similarity(old.text, candidate.text)) for idx, candidate in enumerate(after)), key=lambda x: x[1])
        else:
            idx, score = -1, 0.0
        if score >= 0.43:
            new = after[idx]
            same_unit = _similarity(old.unit, new.unit) >= 0.65
            status = "preserved" if same_unit and score >= 0.72 else ("moved" if not same_unit else "changed")
            result.append(FunctionMatch(old, new, round(score, 2), status))
            matched_after.add(idx)
        else:
            result.append(FunctionMatch(old, None, round(score, 2), "lost"))
    for idx in sorted(set(range(len(after))) - matched_after):
        result.append(FunctionMatch(None, after[idx], 0.0, "new"))
    return result


def _is_generic_duty(text: str) -> bool:
    normalized = _norm(text)
    return any(re.search(pattern, normalized) for pattern in (
        r"прочих поручени",
        r"организу\w* работу (?:днм|дккм|департамент)",
        r"повышени\w* профессионального уровня",
        r"по всему кругу вопросов",
        r"участву\w* в разработке (?:внд|проектов документации)",
    ))


def _find_duplicates(functions: list[Function]) -> list[Finding]:
    findings: list[Finding] = []
    for i, left in enumerate(functions):
        if _is_generic_duty(left.text):
            continue
        for right in functions[i + 1:]:
            if _is_generic_duty(right.text):
                continue
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
    changes = _match_units(before_units, after_units, before_functions, after_functions)
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
