from __future__ import annotations

from collections import Counter

from .models import AnalysisResult


def conclusion(result: AnalysisResult, before_documents: int, after_documents: int) -> str:
    units = Counter(change.status for change in result.unit_changes)
    functions = Counter(match.status for match in result.function_matches)
    findings = Counter(finding.kind for finding in result.findings)
    matched = functions["preserved"] + functions["moved"] + functions["changed"]
    return (
        f"Сопоставлено документов: до — {before_documents}, после — {after_documents}. "
        f"Подразделения: сохранено {units['preserved']}, преобразовано {units['transformed']}, "
        f"исключено {units['removed']}, создано {units['created']}. "
        f"Функции: сопоставлено {matched}, возможная потеря — {functions['lost']}, "
        f"новых — {functions['new']}. "
        f"Потенциальных дублирований — {findings['duplicate']}, "
        f"конфликтов ролей — {findings['conflict']}. "
        "Выводы требуют проверки ответственным сотрудником. "
        "Отсутствие функции в результатах означает отсутствие подтверждённого закрепления "
        "в предоставленных документах, а не доказанное отсутствие функции в деятельности организации."
    )
