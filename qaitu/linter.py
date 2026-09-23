"""Conservative local checks on one edition, using the shared evidence model.

No fuzzy renumbering or legal interpretation. Semantic references are candidates,
not confirmed errors. Continuations, tables of contents and external references
must not be mistaken for duplicate clauses or missing internal targets.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .confidence import assessment
from .models import AnalysisResult, Document, Finding, Fragment


LINTER_VERSION = "local-lint-v1"
IMPLEMENTED_RULES = {
    "LNT-REF": "Ссылка на не найденный в этой редакции пункт",
    "LNT-EMPTY": "Пустой нумерованный пункт",
    "LNT-NUM": "Повтор или разрыв нумерации",
    "LNT-REF-SEM": "Возможное несоответствие ссылки типу нормы",
}
LIMITATIONS = (
    "Проверены только явно напечатанные номера. Оглавление исключено. "
    "Отсутствие пункта в извлечённом тексте не доказывает ошибку оригинала. "
    "Смысловые ссылки — кандидаты, не подтверждённые нарушения. "
    "Проверка сокращений (LNT-ABBR), противоречий (LNT-CONTRA) и автоматические правки пока не реализованы."
)
NUMBER = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3})*)[.)](?:\s*|$)")
REFERENCE = re.compile(
    r"\b(?:пп?\.|пункт(?:а|ам|ах|ами|ы|ов|е|у|ом)?\s+)\s*"
    r"(\d{1,2}(?:\.\d{1,3})*(?:\s*(?:,|и|или|–|—|-)\s*\d{1,2}(?:\.\d{1,3})*)*)",
    re.I,
)
EXTERNAL = re.compile(r"\b(?:закон\w*|кодекс\w*|договор\w*|приказ\w*|устав\w*|стандарт\w*)", re.I)


def _toc_line(text: str) -> bool:
    letters = "".join(re.findall(r"[А-Яа-яЁёA-Za-z]", text))
    return bool(letters and letters.isupper() and re.search(r"\s\d{1,3}\s*$", text))


def clause_index(document: Document) -> dict[str, list[Fragment]]:
    clauses: dict[str, list[Fragment]] = defaultdict(list)
    for source in document.fragments:
        if source.text.strip().casefold() == "оглавление":
            break
        match = NUMBER.match(source.text)
        if match and not _toc_line(source.text):
            clauses[match[1]].append(source)
    return dict(clauses)


def _norm_context(number: str, clauses: dict[str, list[Fragment]]) -> tuple[str, list[Fragment]]:
    parts = number.split(".")
    for size in range(len(parts), 0, -1):
        sources = clauses.get(".".join(parts[:size]), [])
        if len(sources) != 1:
            continue
        text = sources[0].text.casefold()
        if re.search(r"не\s+име\w*\s+прав|запрещ", text):
            return "prohibition", sources
        if re.search(r"име\w*\s+прав|вправе", text):
            return "right", sources
    return "unknown", []


def _finding(code: str, title: str, explanation: str, sources: list[Fragment], score=.85) -> Finding:
    sources = list({s.id: s for s in sources}.values())
    value = assessment(score, method=LINTER_VERSION,
        reasons=[explanation], evidence=sources,
        limitations=["Проверьте оригинал и контекст; автоматическая правка не выполняется."],
        metrics={"rule": code})
    return Finding("hygiene", title, explanation, value.score, sources,
        "Сверить указанные пункты с оригиналом и зафиксировать решение в чек-листе.",
        assessment=value, code=code)


def lint_document(document: Document) -> list[Finding]:
    clauses = clause_index(document)
    findings = []
    for number, sources in clauses.items():
        if len(sources) > 1:
            findings.append(_finding("LNT-NUM", f"Номер {number} встречается несколько раз",
                f"Найдено {len(sources)} явно пронумерованных пунктов с номером {number}. Оглавление не учитывалось.", sources))
        for source in sources:
            body = NUMBER.sub("", source.text, count=1)
            if not re.search(r"[А-Яа-яЁёA-Za-z0-9]", body):
                findings.append(_finding("LNT-EMPTY", f"Пустой пункт {number}",
                    f"После номера {number} в извлечённом тексте нет содержательного текста.", [source], .95))
    # Only internal gaps between observed siblings; do not assume a fragment
    # starts at .1, or invent missing sections at either edge of a partial file.
    siblings: dict[str, list[int]] = defaultdict(list)
    for number in clauses:
        if "." in number:
            parent, child = number.rsplit(".", 1)
            siblings[parent].append(int(child))
    for parent, children in siblings.items():
        ordered = sorted(set(children))
        for left, right in zip(ordered, ordered[1:]):
            if right > left + 1:
                missing = f"{parent}.{left + 1}" if right == left + 2 else f"{parent}.{left + 1}–{parent}.{right - 1}"
                findings.append(_finding("LNT-NUM", f"Разрыв нумерации: {missing}",
                    f"Между явно найденными пунктами {parent}.{left} и {parent}.{right} номера не обнаружены; возможно неполное извлечение.",
                    clauses[f"{parent}.{left}"] + clauses[f"{parent}.{right}"], .7))
    seen = set()
    for source in document.fragments:
        if _toc_line(source.text):
            continue
        for match in REFERENCE.finditer(source.text):
            suffix = source.text[match.end():match.end() + 85]
            # External targets are not resolvable within this document. Prefer
            # skipping ambiguous references over claiming a missing legal norm.
            if EXTERNAL.search(suffix) and not re.match(r"\s*(?:настоящего\s+)?[Пп]оложени", suffix):
                continue
            targets = re.findall(r"\d{1,2}(?:\.\d{1,3})*", match[1])
            if re.search(r"[–—-]", match[1]) and len(targets) == 2:
                first, last = [s.rsplit(".", 1) for s in targets]
                if len(first) == len(last) == 2 and first[0] == last[0] and 0 < int(last[1]) - int(first[1]) <= 100:
                    targets = [f"{first[0]}.{i}" for i in range(int(first[1]), int(last[1]) + 1)]
            missing = [target for target in targets if target not in clauses]
            if missing and (source.id, "missing", tuple(missing)) not in seen:
                seen.add((source.id, "missing", tuple(missing)))
                findings.append(_finding("LNT-REF", "Цель ссылки не найдена: " + ", ".join(missing),
                    "В этой редакции нет явно распознанного пункта: " + ", ".join(missing) + ". Проверьте полноту документа и номера в оригинале.", [source], .8))
            prefix = source.text[max(0, match.start() - 110):match.start()].casefold()
            expected = "prohibition" if re.search(r"не\s+противоречащ\w*\s*$", prefix) else "right" if re.search(r"прав\w*[^.;]{0,50}(?:предусмотр|установ|закрепл)\w*[^.;]*$", prefix) else ""
            if not expected:
                continue
            conflicting, evidence = [], []
            for target in targets:
                actual, context = _norm_context(target, clauses)
                if actual != "unknown" and actual != expected:
                    conflicting.append(target)
                    evidence.extend(clauses[target] + context)
            if conflicting and (source.id, "semantic") not in seen:
                seen.add((source.id, "semantic"))
                findings.append(_finding("LNT-REF-SEM", "Проверить смысл ссылки: " + ", ".join(conflicting),
                    "Фраза перед ссылкой указывает на " + ("ограничения" if expected == "prohibition" else "права") +
                    ", а заголовок целевых пунктов задаёт другой тип нормы. Это эвристический кандидат: ссылка может быть намеренной; требуется проверка человеком.",
                    [source, *evidence], .6))
    return findings


def lint_documents(documents: list[Document]) -> list[Finding]:
    # Separate editions/documents: same clause numbers in two files are normal.
    unique = {}
    for document in documents:
        for finding in lint_document(document):
            key = (finding.code, finding.title, tuple(s.id for s in finding.sources))
            unique[key] = finding
    return list(unique.values())


def inspect_document_pack(documents: list[Document], *, after_complete=False) -> AnalysisResult:
    """Single-edition mode: do not invent a comparison with an empty past."""
    if not documents or any(d.period != "after" for d in documents):
        raise ValueError("Для проверки проекта нужны документы одной редакции в поле «После изменений».")
    warnings = [w for d in documents for w in d.warnings]
    for document in documents:
        if not clause_index(document):
            warnings.append(f"В документе «{document.name}» нет явно распознанной нумерации; ссылки и последовательность пунктов не проверены.")
    return AnalysisResult([], [], [], warnings=warnings,
        sources=list({s.id: s for d in documents for s in d.fragments}.values()),
        coverage={"documents_before": 0, "documents_after": len(documents),
                  "fragments_before": 0, "fragments_after": sum(len(d.fragments) for d in documents)},
        analysis_context={"single_document_review": True, "after_complete_user_declared": after_complete,
                          "linter_version": LINTER_VERSION},
        document_checks=lint_documents(documents))
