#!/usr/bin/env python3
"""Local acceptance smoke check for the supplied internal-audit editions 8 and 9.

No input content is written or sent over the network. These expectations are a
manual sample, not an official benchmark and not a complete assessment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qaitu.analyzer import analyze_documents
from qaitu.extractors import extract_document


DEPARTMENTS = {
    "ДНМ": "непрерывного мониторинга",
    "ДККМ": "контроля качества аудита",
    "ДИТААД": "ит-аудита и анализа данных",
    "ДОА": "операционного аудита",
}


def alias(unit: str) -> str | None:
    value = unit.casefold()
    for abbreviation, words in DEPARTMENTS.items():
        if words in value or re.search(rf"\b{abbreviation.casefold()}\b", value):
            return abbreviation
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Редакция 8, протокол №13 от 2021 года")
    parser.add_argument("--after", type=Path, required=True, help="Редакция 9, протокол №7 от 2022 года")
    args = parser.parse_args()
    documents = []
    for path, period in [(args.before, "before"), (args.after, "after")]:
        with path.open("rb") as handle:
            documents.append(extract_document(handle, path.name, period))
    result = analyze_documents([documents[0]], [documents[1]])
    checks: list[tuple[str, bool]] = []

    def check(label: str, ok: bool) -> None:
        checks.append((label, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {label}")

    before_units = {alias(unit) for unit in result.units_before} - {None}
    after_units = {alias(unit) for unit in result.units_after} - {None}
    check("R02: в старой структуре ДНМ и ДККМ", before_units == {"ДНМ", "ДККМ"})
    check("R02: в новой структуре четыре департамента", after_units == set(DEPARTMENTS))
    created = {alias(change.after or "") for change in result.unit_changes if change.status == "created"}
    check("R02: добавлены только ДИТААД и ДОА", created == {"ДИТААД", "ДОА"})
    preserved = {alias(change.before or "") for change in result.unit_changes if change.status == "preserved"}
    check("R02: ДНМ и ДККМ сохранены", {"ДНМ", "ДККМ"} <= preserved)

    # Same atomic wording, different owner heading; the row must retain both
    # after owners instead of greedily consuming one equivalent assignment.
    transfer_rows = [row for row in result.matrix_rows if any(
        "недостаточн" in f.text.casefold() and "покрыт" in f.text.casefold()
        for f in row.before
    )]
    check("R03: обнаружен перенос взаимодействия с СВК", any(
        "ДНМ" in {alias(f.unit) for f in row.before}
        and {"ДИТААД", "ДОА"} <= {alias(f.unit) for f in row.after}
        and row.status != "lost" for row in transfer_rows
    ))
    reports = [row for row in result.matrix_rows if any(
        "готовит отчеты об итогах выполнения плана" in f.text.casefold()
        and alias(f.unit) == "ДККМ" for f in row.before
    )]
    check("R13: отчеты ДККМ имеют соответствие в новой редакции", any(
        any(alias(f.unit) == "ДККМ" for f in row.after) and row.status != "lost"
        for row in reports
    ))
    check("R13: квартальная и годовая отчетность сохранена в общем контексте", any(
        row.before and any("ежеквартальной" in f.text.casefold() and "итогам года" in f.text.casefold()
                           for f in row.after)
        and any("ежеквартальной" in f.text.casefold() and "итогам года" in f.text.casefold()
                for f in row.before)
        for row in result.matrix_rows
    ))
    right_rows = [row for row in result.matrix_rows if any(
        re.search(r"\b5\.8\.(?:[1-9]|1[0-4])\.", f.source.text)
        and f.norm_type == "right" for f in row.before
    )]
    check("R05: права работников извлечены", len(right_rows) >= 14)
    check("R05: права работников не потеряны из-за перенумерации", bool(right_rows) and all(
        row.after and row.status != "lost" for row in right_rows
    ))
    split_rows = [row for row in result.matrix_rows if any(
        "организация работы проектной команды по проверке" in f.text.casefold()
        for f in row.before
    )]
    check("R15: составная обязанность связана с двумя новыми пунктами", any(
        row.status != "lost" and {"5.3.4", "5.3.5б"} <= {
            reference for f in row.after
            for reference in ("5.3.4", "5.3.5б") if f"п. {reference}" in f.source.locator
        }
        for row in split_rows
    ))
    check("R15: согласование результатов не помечено потерянным", any(
        row.status != "lost" and any("п. 5.3.5в" in f.source.locator for f in row.after)
        for row in result.matrix_rows if any("обсуждение и согласование результатов проверок" in f.text.casefold() for f in row.before)
    ))
    check("R15: разработка документации сопоставлена с ВНД", any(
        row.status != "lost" and any("п. 5.3.12" in f.source.locator for f in row.after)
        for row in result.matrix_rows if any("проектов документации" in f.text.casefold() for f in row.before)
    ))
    all_functions = [f for row in result.matrix_rows for f in row.before + row.after]
    check("R15: титул и определения не являются функциями", not any(
        f.source.text.startswith("УТВЕРЖДЕНО") or "п. 14" in f.source.locator for f in all_functions
    ))
    check("R15: явный руководитель БВА установлен", any(
        "п. 1.4" in f.source.locator and f.unit == "Главный аудитор"
        for f in all_functions
    ))
    known = {fragment.id for document in documents for fragment in document.fragments}
    check("R14: выводы и назначения ссылаются на загруженные источники", all(
        source.id in known for row in result.matrix_rows for function in row.before + row.after
        for source in (function.source, *function.context_sources)
    ) and all(finding.sources and all(source.id in known for source in finding.sources)
              for finding in result.findings))
    print(f"\nПроверено {len(checks)} условий: {sum(ok for _, ok in checks)} успешно.")
    print("Проверка локальная; не измеряет точность всех выводов и не заменяет экспертную оценку.")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
