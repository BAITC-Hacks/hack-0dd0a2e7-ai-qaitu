"""Behavioral regressions for organizational regulations; all fixtures are synthetic."""
import unittest

from qaitu.analyzer import analyze_documents, extract_units_and_functions
from qaitu.extractors import document_from_lines


def document(lines, period="before", name=None):
    return document_from_lines(name or f"synthetic-{period}.docx", period, lines)


def functions_with(functions, text):
    return [f for f in functions if text.casefold() in f.text.casefold()]


def rows_with(result, text):
    return [row for row in result.matrix_rows if any(
        text.casefold() in f.text.casefold() for f in row.before + row.after
    )]


CATALOG = [
    "3.4. В структуру БВА входят следующие департаменты:",
    "а. Департамент методологии (ДМ);",
    "б. Департамент контроля качества (ДКК).",
]


class ContextExtractionRegressionTest(unittest.TestCase):
    def test_staffing_list_job_titles_are_not_functions(self):
        source = document(CATALOG + [
            "3.7. Директору ДМ подчиняются работники в составе следующих должностей:",
            "а. Руководитель направления непрерывного мониторинга.",
            "б. Директор проектов ДМ.",
            "5.4. Директор ДМ:",
            "5.4.1. руководит подготовкой методики внутреннего аудита.",
        ])
        _, functions = extract_units_and_functions([source])
        self.assertFalse(any("Руководитель направления" in f.text for f in functions))
        self.assertFalse(any("Директор проектов" in f.text for f in functions))
        self.assertTrue(any("руководит подготовкой" in f.text for f in functions))

    def test_alias_and_inflected_heading_keep_owner_and_heading_evidence(self):
        source = document(CATALOG + [
            "5.4. Директор департамента методологии (ДМ):",
            "5.4.1. готовит отчеты о выполнении плана работы;",
            "5.4.2. осуществляет управление аудитом по направлениям в зоне ответственности.",
        ])
        units, functions = extract_units_and_functions([source])
        reports = functions_with(functions, "готовит отчеты")
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].owner_known)
        self.assertIn("методолог", reports[0].unit.casefold())
        self.assertFalse(any("управление аудитом" in unit.casefold() for unit in units))
        self.assertTrue(any("Директор" in context.text for context in reports[0].context_sources))

    def test_multiple_owners_in_one_heading_are_not_dropped(self):
        source = document(CATALOG + [
            "5.3. Директоры департаментов ДМ и ДКК:",
            "5.3.1. готовят предложения для включения в план работ БВА;",
        ])
        _, functions = extract_units_and_functions([source])
        proposals = functions_with(functions, "готовят предложения")
        self.assertEqual(len({f.unit for f in proposals}), 2)
        self.assertTrue(all(f.owner_known for f in proposals))
        self.assertEqual(len({f.source.id for f in proposals}), 1)

    def test_unresolved_explicit_department_alias_is_not_assigned_to_all_departments(self):
        source = document(CATALOG + [
            "5.3. Директоры департаментов ДНОВЫЙ:",
            "5.3.1. готовят предложения в план аудита.",
        ])
        _, functions = extract_units_and_functions([source])
        proposals = functions_with(functions, "готовят предложения")
        self.assertTrue(proposals)
        self.assertTrue(all(not f.owner_known for f in proposals))

    def test_explicit_owner_replaces_previous_owner_evidence(self):
        source = document(CATALOG + [
            "5.3. Директор ДМ:",
            "5.3.1. готовит предложения в план аудита.",
            "5.3.2. Департамент контроля качества готовит отчеты о результатах работы.",
        ])
        _, functions = extract_units_and_functions([source])
        reports = functions_with(functions, "готовит отчеты")
        self.assertEqual(len(reports), 1)
        self.assertIn("контроля качества", reports[0].unit.casefold())
        self.assertFalse(any("Директор ДМ" in c.text for c in reports[0].context_sources))

    def test_subclauses_inherit_action_and_owner(self):
        source = document(CATALOG + [
            "5.4. Директор ДМ:",
            "5.4.4. взаимодействует с субъектами внутреннего контроля в части:",
            "а. использования результатов работы других субъектов внутреннего контроля;",
            "б. выявления рисков, имеющих недостаточное покрытие.",
        ])
        _, functions = extract_units_and_functions([source])
        risks = functions_with(functions, "недостаточное покрытие")
        self.assertTrue(risks)
        self.assertTrue(all(f.owner_known and "методолог" in f.unit.casefold() for f in risks))
        self.assertTrue(any("взаимодействует" in c.text for f in risks for c in f.context_sources))

    def test_prohibition_is_inherited_and_not_used_as_execution(self):
        source = document([
            "3.1. Департамент аудита закупок (ДАЗ).",
            "5.1. Директор ДАЗ:",
            "5.1.1. проверяет закупочные процедуры компании.",
            "5.2. Работники ДАЗ не имеют права:",
            "5.2.1. осуществлять закупочные процедуры компании;",
        ], "after")
        _, functions = extract_units_and_functions([source])
        prohibited = functions_with(functions, "осуществлять закупочные")
        self.assertTrue(prohibited)
        self.assertTrue(all(f.norm_type == "prohibition" for f in prohibited))
        before = document([f.text for f in source.fragments], "before")
        result = analyze_documents([before], [source])
        self.assertFalse(any(f.kind == "conflict" for f in result.findings))

    def test_explicit_letter_action_overrides_generic_intro_role_and_negation(self):
        source = document(CATALOG + [
            "5.1. Директор ДКК:",
            "5.1.1. выполняет следующие действия:",
            "а. проверяет закупочные процедуры;",
            "б. не осуществляет закупочные процедуры.",
        ])
        _, functions = extract_units_and_functions([source])
        checks = functions_with(functions, "проверяет закупочные")
        prohibited = functions_with(functions, "не осуществляет закупочные")
        self.assertTrue(checks and prohibited)
        self.assertTrue(all(f.norm_type == "duty" and f.role == "control" for f in checks))
        self.assertTrue(all(f.norm_type == "prohibition" for f in prohibited))

    def test_mixed_positive_and_prohibited_actions_keep_separate_norm_types(self):
        source = document([
            "1.1. Департамент контроля качества проверяет закупочные процедуры; не осуществляет закупочные процедуры.",
        ])
        _, functions = extract_units_and_functions([source])
        checks = functions_with(functions, "проверяет закупочные")
        prohibited = functions_with(functions, "не осуществляет закупочные")
        self.assertTrue(checks and prohibited)
        self.assertTrue(all(f.norm_type == "duty" for f in checks))
        self.assertTrue(all(f.norm_type == "prohibition" for f in prohibited))

    def test_inline_negation_does_not_change_unit_name_or_next_clause(self):
        source = document([
            "1.1. Департамент контроля качества не осуществляет закупочные процедуры.",
            "1.2. проверяет закупочные процедуры.",
        ])
        units, functions = extract_units_and_functions([source])
        self.assertEqual(set(units), {"Департамент контроля качества"})
        checks = functions_with(functions, "проверяет закупочные")
        self.assertTrue(checks)
        self.assertTrue(all(f.norm_type == "duty" for f in checks))

    def test_explicit_negation_is_not_a_positive_duty(self):
        source = document([
            "3.1. Департамент аудита закупок не осуществляет закупочные процедуры компании.",
        ])
        _, functions = extract_units_and_functions([source])
        assignments = functions_with(functions, "закупочные процедуры")
        self.assertFalse(any(f.norm_type == "duty" for f in assignments))

    def test_section_boundary_resets_owner_and_inherited_prohibition(self):
        source = document(CATALOG + [
            "5.1. Работники ДМ не имеют права:",
            "5.1.1. осуществлять закупочные процедуры компании.",
            "6. Порядок подготовки отчетности",
            "6.1. готовит квартальные отчеты о выполнении плана.",
        ])
        _, functions = extract_units_and_functions([source])
        reports = functions_with(functions, "квартальные отчеты")
        self.assertTrue(reports)
        self.assertTrue(all(not f.owner_known for f in reports))
        self.assertTrue(all(f.norm_type == "duty" for f in reports))
        self.assertFalse(any("не имеют права" in c.text for f in reports for c in f.context_sources))

    def test_owner_context_does_not_leak_between_documents(self):
        first = document([
            "3.1. Департамент методологии разрабатывает методику аудита.",
        ], name="one.docx")
        second = document([
            "1.1. готовит квартальные отчеты о выполнении плана.",
        ], name="two.docx")
        _, functions = extract_units_and_functions([first, second])
        reports = functions_with(functions, "квартальные отчеты")
        self.assertTrue(reports, "Неопределенное назначение нужно оставить видимым для проверки")
        self.assertTrue(all(not f.owner_known for f in reports))


class MatrixRegressionTest(unittest.TestCase):
    def test_same_owner_repeated_in_two_documents_has_one_row_and_all_evidence(self):
        line = "1.1. Департамент методологии ведет реестр аудиторских рекомендаций."
        before = [document([line])]
        after = [document([line], "after", "regulation.docx"),
                 document([line], "after", "job-description.docx")]
        result = analyze_documents(before, after)
        rows = rows_with(result, "реестр аудиторских рекомендаций")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len({f.unit for f in rows[0].after}), 1)
        self.assertEqual({f.source.document for f in rows[0].after},
                         {"regulation.docx", "job-description.docx"})
        self.assertFalse(rows[0].candidate_overlap)
        self.assertFalse(any(f.kind == "duplicate" for f in result.findings))

    def test_two_old_owners_consolidating_to_one_does_not_lose_the_function(self):
        before = document([
            "1.1. Департамент методологии ведет реестр аудиторских рекомендаций.",
            "2.1. Служба контроля качества ведет реестр аудиторских рекомендаций.",
        ])
        after = document([
            "1.1. Департамент методологии ведет реестр аудиторских рекомендаций.",
        ], "after")
        result = analyze_documents([before], [after])
        rows = rows_with(result, "реестр аудиторских рекомендаций")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len({f.unit for f in rows[0].before}), 2)
        self.assertEqual(len({f.unit for f in rows[0].after}), 1)
        self.assertNotEqual(rows[0].status, "lost")
        self.assertFalse(any(f.kind == "loss" for f in result.findings))

    def test_one_function_transferred_to_two_departments_stays_one_row(self):
        before = document([
            "1.1. Департамент методологии ведет реестр аудиторских рекомендаций.",
        ])
        after = document([
            "1.1. Департамент операционного аудита ведет реестр аудиторских рекомендаций.",
            "2.1. Департамент ИТ-аудита ведет реестр аудиторских рекомендаций.",
        ], "after")
        result = analyze_documents([before], [after])
        rows = rows_with(result, "реестр аудиторских рекомендаций")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len({f.unit for f in rows[0].after}), 2)
        self.assertNotEqual(rows[0].status, "lost")
        self.assertFalse(any(f.kind == "loss" for f in result.findings))

    def test_renumbered_right_is_not_lost(self):
        before = document(CATALOG + [
            "5.8. Работники ДМ имеют право:",
            "5.8.1. получать документы, необходимые для проведения аудита.",
        ])
        after = document(CATALOG + [
            "5.7. Работники ДМ имеют право:",
            "5.7.1. получать документы, необходимые для проведения аудита.",
        ], "after")
        result = analyze_documents([before], [after])
        rows = rows_with(result, "необходимые для проведения аудита")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].norm_type, "right")
        self.assertTrue(rows[0].before and rows[0].after)
        self.assertNotEqual(rows[0].status, "lost")

    def test_unparsed_after_is_unknown_instead_of_total_loss(self):
        before = document([
            "1.1. Департамент методологии ведет реестр аудиторских рекомендаций.",
        ])
        after = document(["Изображение штатного расписания; доступный текст отсутствует."], "after")
        result = analyze_documents([before], [after])
        rows = rows_with(result, "реестр аудиторских рекомендаций")
        self.assertTrue(rows)
        self.assertEqual(rows[0].status, "unknown")
        self.assertFalse(any(f.kind == "loss" for f in result.findings))
        self.assertTrue(result.warnings)

    def test_all_matrix_and_conclusion_sources_belong_to_input_catalog(self):
        before = document([
            "1.1. Департамент методологии ведет реестр аудиторских рекомендаций.",
            "1.2. Департамент методологии готовит отчеты о выполнении плана.",
        ])
        after = document([
            "1.1. Служба контроля качества ведет реестр аудиторских рекомендаций.",
        ], "after")
        result = analyze_documents([before], [after])
        known = {f.id: f for d in [before, after] for f in d.fragments}
        self.assertEqual({f.id for f in result.sources}, set(known))
        self.assertTrue(result.matrix_rows)
        for row in result.matrix_rows:
            for function in row.before + row.after:
                for source in (function.source, *function.context_sources):
                    self.assertIn(source.id, known)
                    self.assertEqual(source, known[source.id])
        for finding in result.findings:
            self.assertTrue(finding.sources)
            for source in finding.sources:
                self.assertIn(source.id, known)
                self.assertEqual(source, known[source.id])
        for change in result.unit_changes:
            self.assertTrue(change.sources)
            self.assertTrue(all(source.id in known for source in change.sources))


if __name__ == "__main__":
    unittest.main()
