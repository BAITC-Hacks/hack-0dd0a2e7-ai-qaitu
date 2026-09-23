import unittest

from qaitu.analyzer import analyze_documents
from qaitu.analyzer import extract_units_and_functions
from qaitu.analyzer import _find_conflicts
from qaitu.demo import demo_documents
from qaitu.extractors import document_from_lines
from qaitu.models import Function
from qaitu.report import conclusion


class AnalyzerTest(unittest.TestCase):
    def test_demo_detects_required_cases(self):
        before, after = demo_documents()
        result = analyze_documents(before, after)
        kinds = {finding.kind for finding in result.findings}
        self.assertTrue({"loss", "duplicate", "conflict"} <= kinds)
        self.assertTrue(all(finding.sources for finding in result.findings))
        self.assertTrue(any(change.status in {"created", "transformed"} for change in result.unit_changes))

    def test_every_source_is_traceable(self):
        before, after = demo_documents()
        result = analyze_documents(before, after)
        for finding in result.findings:
            for source in finding.sources:
                self.assertTrue(source.id)
                self.assertTrue(source.document)
                self.assertTrue(source.locator)
                self.assertTrue(source.text)

    def test_complete_rename_matches_by_functions(self):
        before = [document_from_lines("before.docx", "before", [
            "Департамент непрерывного мониторинга осуществляет оценку качества обслуживания клиентов.",
            "Департамент непрерывного мониторинга ведет реестр жалоб клиентов.",
        ])]
        after = [document_from_lines("after.docx", "after", [
            "Центр гарантий и аналитики осуществляет оценку качества обслуживания клиентов.",
            "Центр гарантий и аналитики ведет реестр жалоб клиентов.",
        ])]
        result = analyze_documents(before, after)
        self.assertEqual(len(result.unit_changes), 1)
        self.assertEqual(result.unit_changes[0].status, "transformed")
        self.assertEqual(result.unit_changes[0].after, "Центр гарантий и аналитики")

    def test_conclusion_uses_actual_counts(self):
        before, after = demo_documents()
        text = conclusion(analyze_documents(before, after), len(before), len(after))
        self.assertIn("возможная потеря — 1", text)
        self.assertIn("конфликтов ролей — 1", text)
        self.assertIn("а не доказанное отсутствие", text)

    def test_real_provision_excerpts_keep_duties_with_their_owner(self):
        before = document_from_lines("Положение 2021", "before", [
            "3.4. БВА состоит из следующих структурных подразделений:",
            "а. Департамент непрерывного мониторинга системы внутреннего контроля (ДНМ).",
            "б. Департамент контроля качества аудита и методологии (ДККМ).",
            "5.4. Директор департамента непрерывного мониторинга системы внутреннего контроля:",
            "5.4.5. организует содействие в разработке процедур и мероприятий по совершенствованию СУР, ВК и КУ.",
            "5.5. Директор департамента контроля качества аудита и методологии:",
            "5.5.2. организует непрерывный мониторинг качества деятельности внутреннего аудита.",
        ])
        after = document_from_lines("Положение 2022", "after", [
            "3.4. БВА состоит из следующих структурных подразделений:",
            "а. Департамент ИТ-аудита и анализа данных (ДИТААД).",
            "б. Департамент операционного аудита (ДОА).",
            "в. Департамент непрерывного мониторинга системы внутреннего контроля (ДНМ).",
            "г. Департамент контроля качества аудита и методологии (ДККМ).",
            "5.4. Директор департамента непрерывного мониторинга системы внутреннего контроля:",
            "5.4.4. организует содействие в разработке процедур и мероприятий по совершенствованию СУР, ВК и КУ.",
            "5.5. Директор департамента контроля качества аудита и методологии:",
            "5.5.2. организует непрерывный мониторинг качества деятельности внутреннего аудита.",
        ])
        units, functions = extract_units_and_functions([before])
        self.assertEqual(len(units), 2)
        self.assertEqual(len(functions), 2)
        self.assertIn("непрерывного мониторинга системы внутреннего контроля", functions[0].unit)
        self.assertIn("контроля качества аудита и методологии", functions[1].unit)
        self.assertIn("п. 5.4.5", functions[0].source.locator)
        result = analyze_documents([before], [after])
        self.assertEqual({change.status for change in result.unit_changes}, {"preserved", "created"})
        self.assertEqual(sum(finding.kind == "loss" for finding in result.findings), 0)

    def test_shared_administrative_duties_are_not_reported_as_duplicates(self):
        before = [document_from_lines("before", "before", [
            "Отдел А проверяет закупочные процедуры.",
        ])]
        after = [document_from_lines("after", "after", [
            "Отдел А осуществляет выполнение прочих поручений Главного аудитора.",
            "Отдел Б осуществляет выполнение прочих поручений Главного аудитора.",
        ])]
        result = analyze_documents(before, after)
        self.assertEqual(sum(finding.kind == "duplicate" for finding in result.findings), 0)

    def test_repeated_old_duty_is_not_reported_lost_when_one_new_owner_remains(self):
        before = [document_from_lines("before", "before", [
            "Отдел А ведет реестр аудиторских рекомендаций.",
            "Отдел Б ведет реестр аудиторских рекомендаций.",
        ])]
        after = [document_from_lines("after", "after", [
            "Управление В ведет реестр аудиторских рекомендаций.",
        ])]
        result = analyze_documents(before, after)
        self.assertEqual(sum(match.status == "lost" for match in result.function_matches), 0)

    def test_corporate_wide_duties_do_not_create_role_conflicts(self):
        source = document_from_lines("after", "after", ["Контроль и выполнение процесса закупок."]).fragments[0]
        duties = [
            Function("1", "БВА (общие функции)", "контролирует процедуры закупок и риски закупок", source, role="control"),
            Function("2", "БВА (общие функции)", "выполняет процедуры закупок и оценку рисков закупок", source, role="execute"),
        ]
        self.assertEqual(_find_conflicts(duties), [])

    def test_compound_duty_is_linked_to_two_new_clauses(self):
        before = [document_from_lines("old.docx", "before", [
            "3.4. БВА состоит из следующих структурных подразделений:",
            "а. Департамент внутреннего аудита (ДВА).",
            "5.3. Директор ДВА:",
            "5.3.1. организация работы проектной команды по проверке и организация контроля качества работы проектной команды.",
        ])]
        after = [document_from_lines("new.docx", "after", [
            "3.4. БВА состоит из следующих структурных подразделений:",
            "а. Департамент операционного аудита (ДОА).",
            "5.3. Директор ДОА:",
            "5.3.1. организует работу проектных команд в зоне ответственности.",
            "5.3.2. контроль качества работы проектной команды.",
        ])]
        result = analyze_documents(before, after)
        row = next(row for row in result.matrix_rows if "организация работы проектной команды" in row.label)
        self.assertEqual(row.status, "moved")
        self.assertEqual({f.source.locator for f in row.after}, {"строка 4, п. 5.3.1", "строка 5, п. 5.3.2"})
        self.assertFalse(any(f.kind == "loss" and f.matrix_row_id == row.id for f in result.findings))

    def test_discussion_agreement_and_vnd_rewording_are_not_lost(self):
        before = [document_from_lines("old.docx", "before", [
            "3.4. БВА состоит из следующих структурных подразделений:",
            "а. Департамент внутреннего аудита (ДВА).",
            "5.3. Директор ДВА:",
            "5.3.1. обсуждение и согласование результатов проверок с Руководителями Общества по проверяемым направлениям.",
            "5.3.2. участвует в разработке проектов документации, регламентирующей работу БВА.",
        ])]
        after = [document_from_lines("new.docx", "after", [
            "3.4. БВА состоит из следующих структурных подразделений:",
            "а. Департамент операционного аудита (ДОА).",
            "5.3. Директор ДОА:",
            "5.3.1. согласование результатов проверок с Руководителями Общества по проверяемым направлениям.",
            "5.3.2. участвуют в разработке ВНД БВА.",
        ])]
        result = analyze_documents(before, after)
        for phrase in ("обсуждение и согласование", "проектов документации"):
            row = next(row for row in result.matrix_rows if phrase in row.label)
            self.assertTrue(row.after, phrase)
            self.assertNotEqual(row.status, "lost")

    def test_cover_is_not_a_duty_and_explicit_chief_auditor_is_owner(self):
        document = document_from_lines("regulation.pdf", "before", [
            "УТВЕРЖДЕНО Советом директоров Протокол No 7 ПОЛОЖЕНИЕ О ВНУТРЕННЕМ АУДИТЕ",
            "1.4. Руководство БВА осуществляет Главный аудитор в соответствии с Уставом Общества.",
        ])
        _, functions = extract_units_and_functions([document])
        self.assertFalse(any("УТВЕРЖДЕНО" in function.source.text for function in functions))
        leadership = [function for function in functions if "Руководство БВА" in function.source.text]
        self.assertTrue(leadership)
        self.assertTrue(all(function.unit == "Главный аудитор" for function in leadership))

    def test_definitions_describe_terms_not_assignments(self):
        document = document_from_lines("regulation.docx", "before", [
            "5.1. Главный аудитор организует работу БВА.",
            "14. Термины и определения",
            "Риск — потенциальное событие, которое влияет на цели Общества.",
            "Непрерывный аудит — метод, который обеспечивает оценку рисков и проверку данных.",
        ])
        _, functions = extract_units_and_functions([document])
        self.assertTrue(any("организует работу" in function.text for function in functions))
        self.assertFalse(any("потенциальное событие" in function.source.text or "Непрерывный аудит" in function.source.text for function in functions))


if __name__ == "__main__":
    unittest.main()
