"""Deterministic monthly demonstration reports; never represent real evidence."""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import Workbook

from qaitu.metrics_catalog import METRIC_CATALOG
from qaitu.metrics_models import Hypothesis, StructuralEvent


def generate_reports(output_dir) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7421)
    files = []
    values_expected = 0
    samples = []
    scopes = [("ДНМ", "DNM"), ("ДККМ", "DKKM"), ("БВА:ИТ-аудит", "BVA_IT_PROCESS")]
    for scope, slug in scopes:
        for year in (2021, 2022, 2023):
            book = Workbook()
            sheet = book.active
            sheet.title = "Показатели"
            sheet.append(["ТЕСТОВЫЕ ДАННЫЕ — синтетический сценарий, не фактические результаты"])
            sheet.append(["Период", "Подразделение", "Метрика", "Значение", "Единица"])
            for month in range(1, 13):
                post = year == 2023
                it_scope = scope == "БВА:ИТ-аудит"
                fte = 8 if it_scope and post else 4 if it_scope else 6
                done = (10 if post else 4) if it_scope else 5
                done += rng.choice((-1, 0, 0, 1))
                planned = done + 1
                delay = (10 if post else 2) if scope == "ДККМ" else 2
                delay += rng.choice((-1, 0, 0, 1))
                values = {
                    "audits_planned": planned, "audits_done": done,
                    "plan_completion": round(100 * done / planned, 2),
                    "findings_total": done * 3, "findings_material": max(1, done // 3),
                    "recs_issued": done * 4, "recs_accepted": 90 + rng.choice((-2, 0, 2)),
                    "recs_implemented": 85 + rng.choice((-2, 0, 2)), "action_overdue": 2,
                    "audit_cycle_days": 28 + rng.choice((-1, 0, 1)), "report_delay_days": delay,
                    "headcount": fte + 1, "fte_actual": fte,
                    "budget_plan": fte * 1100, "budget_fact": fte * 1000,
                    "assurance_coverage": 75 + rng.choice((-1, 0, 1)), "qa_score": 85,
                    "hotline_cases": 8, "hotline_closed": 7,
                }
                for metric, definition in METRIC_CATALOG.items():
                    sheet.append([f"{year}-{month:02d}", scope, metric, values[metric], definition.unit])
                    values_expected += 1
                    samples.append(dict(metric=metric, value=values[metric], unit_scope=scope, period=f"{year}-{month:02d}"))
            sheet.freeze_panes = "D3"
            sheet.column_dimensions["A"].width = 15
            sheet.column_dimensions["B"].width = 24
            sheet.column_dimensions["C"].width = 27
            sheet.column_dimensions["D"].width = 15
            sheet.column_dimensions["E"].width = 15
            metadata = book.create_sheet("_metadata")
            metadata.append(["is_synthetic", True])
            metadata.append(["label", "ТЕСТОВЫЕ ДАННЫЕ"])
            metadata.append(["scenario", "W1 + S1 + HIRE1; continuous IT-process scope"])
            path = output / f"SYNTH_{slug}_{year}.xlsx"
            book.save(path)
            files.append(str(path))
    evidence_file = "SYNTH_manifest.json"
    def evidence(quote):
        return [{"file": evidence_file, "clause_id": "synthetic_scenario", "quote": "ТЕСТОВЫЕ ДАННЫЕ: " + quote}]
    events = [
        StructuralEvent("W1", "func_weakened", ["ДККМ"], "2022-12-23",
                        evidence("В демонстрационном сценарии снят локальный квартальный срок отчётности ДККМ."),
                        finding_id="W1", description="ТЕСТОВЫЕ ДАННЫЕ: ослабление локального срока отчётности",
                        metric_codes=["report_delay_days"], mechanism="Локальная периодичность отчётности влияет на её своевременность; это заданный механизм синтетического сценария.",
                        mechanism_confirmed=True, is_synthetic=True, date_basis="synthetic_scenario"),
        StructuralEvent("S1", "unit_created", ["БВА:ИТ-аудит"], "2022-12-23",
                        evidence("В сценарии создан ДИТААД; сравнивается непрерывный процесс ИТ-аудита БВА до/после, а не несуществовавшее подразделение."),
                        finding_id="S1", description="ТЕСТОВЫЕ ДАННЫЕ: создание ДИТААД; метрики непрерывного процесса ИТ-аудита",
                        metric_codes=["audits_done"], mechanism="Выделение специализированного подразделения может сопровождаться изменением объёма ИТ-проверок.",
                        mechanism_confirmed=True, is_synthetic=True, date_basis="synthetic_scenario"),
        StructuralEvent("HIRE1", "staff_increase", ["БВА:ИТ-аудит"], "2023-01-01",
                        evidence("Фактическая численность процесса ИТ-аудита увеличена с 4 до 8 FTE."),
                        finding_id="HIRE1", description="ТЕСТОВЫЕ ДАННЫЕ: одновременный найм; альтернативное объяснение роста числа проверок",
                        metric_codes=["fte_actual", "audits_done"], mechanism="Рост числа работников может сопровождаться ростом объёма проверок без сопоставимого роста производительности.",
                        mechanism_confirmed=True, is_synthetic=True, date_basis="synthetic_scenario"),
    ]
    hypotheses = [
        Hypothesis("hyp_W1", "Scenario", "W1", "ТЕСТОВЫЕ ДАННЫЕ: после ослабления срока задержка отчётности вырастет", "report_delay_days", "up", "2023-03-31", unit_scope="ДККМ", event_id="W1", is_synthetic=True),
        Hypothesis("hyp_S1", "Scenario", "S1", "ТЕСТОВЫЕ ДАННЫЕ: объём ИТ-проверок вырастет; требуется контроль найма", "audits_done", "up", "2023-03-31", unit_scope="БВА:ИТ-аудит", event_id="S1", is_synthetic=True),
    ]
    controls = {"W1": {"scope": "ДНМ", "verified": True, "reason": "ТЕСТОВЫЕ ДАННЫЕ: ДНМ задан как стабильная незатронутая контрольная группа"},
                "S1": {"scope": "ДНМ", "verified": True, "reason": "ТЕСТОВЫЕ ДАННЫЕ: стабильный контроль для объёма проверок"}}
    manifest = dict(is_synthetic=True, label="ТЕСТОВЫЕ ДАННЫЕ", files=files, paths=files,
                    events=[asdict(item) for item in events], hypotheses=[asdict(item) for item in hypotheses],
                    controls=controls, control_scopes=controls, expected_values=values_expected, expected_samples=samples)
    (output / evidence_file).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Создать только синтетические отчёты для демонстрации")
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--output", dest="output_option", default=None)
    args = parser.parse_args()
    print(json.dumps({key: value for key, value in generate_reports(args.output_option or args.output_dir or "data/synthetic_reports").items() if key != "expected_samples"}, ensure_ascii=False, indent=2))
