import unittest

from qaitu.analyzer import analyze_documents
from qaitu.analyzer import extract_units_and_functions
from qaitu.demo import demo_documents
from qaitu.extractors import document_from_lines
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


if __name__ == "__main__":
    unittest.main()
