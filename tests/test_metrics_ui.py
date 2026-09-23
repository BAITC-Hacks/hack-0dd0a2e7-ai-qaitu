import builtins
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from qaitu.config import OpenAISettings
from qaitu.metrics_models import Hypothesis, IngestionResult, MetricAmbiguity, MetricSource, MetricValue, MetricsRun, PeriodComparison, Recommendation, StructuralEvent
from qaitu.metrics_ui import TEST_LABEL, _chart_rows, value_rows

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def widget(widgets, label):
    return next(item for item in widgets if item.label == label)


def fixture(synthetic=True):
    values = [MetricValue(f"value_{index}", "report_delay_days", float(index + 2), "ДККМ", f"202{2 if index < 2 else 3}-0{index % 2 + 1}-01", f"202{2 if index < 2 else 3}-0{index % 2 + 1}-28", "month",
                          MetricSource(("SYNTH_" if synthetic else "") + "report.xlsx", f"Задержка отчёта: {index + 2} дней", sheet="Свод", cell=f"C{index + 2}"), is_synthetic=synthetic, unit="дни") for index in range(4)]
    event = StructuralEvent("event1", "func_weakened", ["ДККМ"], "2022-12-23", [{"clause_id": "5.5.3", "quote": "Подготовка отчётов об итогах выполнения плана работы."}], description="Периодичность отчёта уточняется", metric_codes=["report_delay_days"], mechanism="Проверить срок подготовки отчётов.", mechanism_confirmed=True, is_synthetic=synthetic)
    hypothesis = Hypothesis("hyp1", "Scenario", "event1", "Задержка может увеличиться", "report_delay_days", "up", "2023-03-31", unit_scope="ДККМ", event_id="event1", is_synthetic=synthetic)
    return IngestionResult(values=values), [event], [hypothesis], {}


class MetricsInterfaceTests(unittest.TestCase):
    def setUp(self):
        patches = [patch.dict(os.environ, {"METRICS_MODULE": "on"}),
                   patch("qaitu.config.load_openai_settings", return_value=OpenAISettings()),
                   patch("qaitu.ai_agent.run_comparison_agent"),
                   patch("qaitu.metrics_service.demo_inputs", side_effect=lambda: deepcopy(fixture())),
                   patch("qaitu.metrics_service.run_metrics", side_effect=self.fake_run),
                   patch("qaitu.metrics_service.save_decision"),
                   patch("qaitu.metrics_service.ingest_reports")]
        self.patches = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)

    @staticmethod
    def fake_run(values, events, *, hypotheses=None, control_scopes=None, use_ai=False, **kwargs):
        if use_ai:
            raise AssertionError("Tests must not enable external API")
        return MetricsRun("test-run", values[0].is_synthetic, values=values, events=events, hypotheses=hypotheses or [], settings={"control_scopes": control_scopes or {}})

    def app(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        return app

    def workspace(self):
        app = self.app()
        widget(app.radio, "Рабочий раздел").set_value("Эффективность и оптимизация").run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        return app

    def test_workspace_is_available_without_core_result_and_synthetic_demo_is_separate(self):
        app = self.workspace()
        self.assertEqual(len(app.tabs), 6)
        widget(app.radio, "Источник показателей").set_value(TEST_LABEL).run()
        widget(app.button, "Запустить демо эффективности").click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertTrue(app.session_state["metrics_demo_run"].is_synthetic)
        self.assertEqual(len(app.session_state["metrics_demo_run"].values), 4)
        self.patches[2].assert_not_called()
        self.patches[6].assert_not_called()
        table = next(item.value for item in app.dataframe if "Данные" in item.value.columns)
        self.assertTrue((table["Данные"] == TEST_LABEL).all())
        chart = app.get("vega_lite_chart")[0]
        self.assertEqual(TEST_LABEL, json.loads(chart.proto.spec)["title"]["text"])
        widget(app.radio, "Источник показателей").set_value("РЕАЛЬНЫЕ ОТЧЁТЫ").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["metrics_real_ingestion"].values, [])

    def test_off_flag_never_imports_metrics_modules_and_core_still_renders(self):
        native_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.startswith("qaitu.metrics"):
                raise AssertionError("Metrics imported while disabled")
            return native_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"METRICS_MODULE": "off"}), patch("builtins.__import__", side_effect=guarded):
            app = self.app()
        self.assertFalse(any(item.label == "Рабочий раздел" for item in app.radio))
        self.assertTrue(any(item.label == "До изменений" for item in app.get("file_uploader")))

    def test_ambiguous_value_requires_human_confirmation_and_keeps_source(self):
        app = self.workspace()
        source = MetricSource("report.xlsx", "Задержка сдачи отчётов 7 дней", sheet="Свод", cell="D5")
        app.session_state["metrics_real_ingestion"] = IngestionResult(ambiguous=[MetricAmbiguity("a1", "Период и владелец требуют подтверждения", source, proposed={"metric": "report_delay_days", "value": 7, "unit_scope": "ДККМ", "period_start": "2023-01-01", "period_end": "2023-01-31", "granularity": "month", "unit": "дни"})])
        app.run()
        widget(app.button, "Сохранить сопоставление").click().run()
        self.assertEqual(len(app.session_state["metrics_real_ingestion"].ambiguous), 1)
        widget(app.checkbox, "Я сверил показатель, число, подразделение и период с источником").check()
        widget(app.button, "Сохранить сопоставление").click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        ingestion = app.session_state["metrics_real_ingestion"]
        self.assertEqual(len(ingestion.ambiguous), 0)
        self.assertEqual(ingestion.values[0].value, 7)
        self.assertEqual(ingestion.values[0].source.cell, "D5")
        self.assertTrue(ingestion.values[0].confirmed_by_human)

    def test_every_synthetic_table_row_is_labeled(self):
        ingestion, *_ = fixture()
        rows = value_rows(ingestion.values)
        self.assertTrue(all(row["Данные"] == TEST_LABEL for row in rows))
        self.assertTrue(all(row["Источник"] and row["Начало"] and row["Конец"] for row in rows))

    def test_control_role_requires_matching_metric_period_and_units(self):
        ingestion, *_ = fixture()
        observed = ingestion.values[0]
        control = deepcopy(observed)
        control.id, control.unit_scope = "control-delay", "ДНМ"
        headcount = deepcopy(control)
        headcount.id, headcount.metric, headcount.unit = "control-headcount", "headcount", "чел."
        comparison = PeriodComparison("cmp", "report_delay_days", "ДККМ", "event", {"granularity": "month"}, {"granularity": "month"}, 1, None, method="diff_in_diff", metric_ids=[observed.id, control.id], control_scope="ДНМ")
        run = MetricsRun("chart", True, values=[observed, control, headcount], comparisons=[comparison])
        self.assertEqual(_chart_rows(run, "headcount", "month", "чел.")[0]["Роль"], "Наблюдаемое подразделение")
        self.assertEqual(_chart_rows(run, "report_delay_days", "month", "дни")[1]["Роль"], "Контрольная группа")
        comparison.before["granularity"] = comparison.after["granularity"] = "year"
        self.assertTrue(all(row["Роль"] == "Наблюдаемое подразделение" for row in _chart_rows(run, "report_delay_days", "month", "дни")))
        comparison.before["granularity"] = comparison.after["granularity"] = "month"
        control.unit = "часы"
        self.assertEqual(_chart_rows(run, "report_delay_days", "month", "часы")[0]["Роль"], "Наблюдаемое подразделение")

    def test_chart_separates_months_years_and_currencies(self):
        ingestion, *_ = fixture(synthetic=False)
        monthly = ingestion.values[0]
        monthly.metric, monthly.unit = "budget_fact", "KZT"
        annual = deepcopy(monthly)
        annual.id, annual.granularity, annual.value = "annual", "year", 120
        dollars = deepcopy(monthly)
        dollars.id, dollars.unit, dollars.value = "dollars", "USD", 30
        second_month = deepcopy(monthly)
        second_month.id, second_month.period_end = "month2", "2022-02-28"
        values = [monthly, second_month, annual, dollars]
        run = MetricsRun("mixed-units", False, values=values)
        app = self.workspace()
        app.session_state["metrics_real_ingestion"] = IngestionResult(values=values)
        app.session_state["metrics_real_run"] = run
        app.run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertEqual(widget(app.selectbox, "Детализация периода").value, "month")
        self.assertEqual(widget(app.selectbox, "Единица измерения на графике").value, "KZT")
        self.assertEqual(len(_chart_rows(run, "budget_fact", "month", "KZT")), 2)
        self.assertEqual(_chart_rows(run, "budget_fact", "month", "USD")[0]["Значение"], 30)
        self.assertEqual(_chart_rows(run, "budget_fact", "year", "KZT")[0]["Значение"], 120)
        widget(app.selectbox, "Единица измерения на графике").set_value("USD").run()
        self.assertFalse(app.exception)
        widget(app.selectbox, "Детализация периода").set_value("year").run()
        self.assertFalse(app.exception)
        self.assertFalse(any(item.label == "Единица измерения на графике" for item in app.selectbox))

    def test_recommendation_buttons_store_a_human_decision(self):
        app = self.workspace()
        ingestion, events, hypotheses, controls = fixture(synthetic=False)
        recommendation = Recommendation("rec1", "Уточнить периодичность отчётов", "restore_function", {"attributions": ["att1"]}, {"metric": "report_delay_days", "direction": "down"}, "medium", ["Нужно проверить ресурсы"])
        run = MetricsRun("decision-run", False, values=ingestion.values, events=events, recommendations=[recommendation])
        app.session_state["metrics_real_ingestion"] = ingestion
        app.session_state["metrics_real_run"] = run
        app.run()
        widget(app.button, "Принять").click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.patches[5].assert_called_once()
        args = self.patches[5].call_args.args
        self.assertEqual(args[1:], ("rec1", "accept"))
        self.assertEqual(app.session_state["metrics_real_run"].recommendations[0].decision, "accept")


if __name__ == "__main__":
    unittest.main()
