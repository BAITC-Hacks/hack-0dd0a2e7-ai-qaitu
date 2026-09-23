"""Arithmetic and attribution contracts; all fixtures are synthetic."""
import calendar
from dataclasses import asdict, replace
from datetime import date
import json
import re
import tempfile
import unittest
from pathlib import Path

from qaitu.metrics_analysis import (attribute_comparisons, calculate_confidence, check_hypotheses,
    compare_metrics, events_from_core, hypothesis_accuracy, recommend_actions)
from qaitu.metrics_catalog import METRIC_CATALOG
from qaitu.metrics_models import (Hypothesis, KnowledgeRecord, MetricSource, MetricValue, StructuralEvent)


def value(year, month, number, metric='report_delay_days', scope='ДККМ', **kwargs):
    start, end = f'{year}-{month:02d}-01', f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]}'
    key = f'{scope}:{metric}:{year}-{month:02d}'
    defaults = dict(id=key, metric=metric, value=number, unit_scope=scope, period_start=start,
        period_end=end, granularity='month', source=MetricSource('SYNTH_report.xlsx', f'{metric}: {number}', sheet='Данные', cell='D4'),
        is_synthetic=True, confirmed_by_human=True, unit=METRIC_CATALOG[metric].unit)
    defaults.update(kwargs)
    return MetricValue(**defaults)


def year_value(year, number, metric='report_delay_days', scope='ДККМ'):
    return value(year, 1, number, metric, scope, period_start=f'{year}-01-01', period_end=f'{year}-12-31', granularity='year', id=f'{scope}:{metric}:{year}:year')


def event(event_id='W1', scope='ДККМ', metric='report_delay_days', **kwargs):
    defaults = dict(id=event_id, type='func_weakened', units=[scope], effective_date='2022-12-23',
        evidence=[{'file': 'SYNTH.docx', 'clause_id': '5.5.3', 'quote': 'ТЕСТОВЫЕ ДАННЫЕ: изменен локальный срок отчетности.'}],
        description='ТЕСТОВЫЕ ДАННЫЕ: изменение локального срока', metric_codes=[metric],
        mechanism='Сопоставление локальной периодичности и задержки отчетов.', mechanism_confirmed=True, is_synthetic=True)
    defaults.update(kwargs)
    return StructuralEvent(**defaults)


def basic_values(metric='report_delay_days', scope='ДККМ', before=2, after=10, n_before=2, n_after=2):
    return [value(2022, month, before, metric, scope) for month in range(1, n_before + 1)] + [value(2023, month, after, metric, scope) for month in range(1, n_after + 1)]


class MetricsArithmeticTests(unittest.TestCase):
    def test_crossing_and_boundary_periods_are_excluded(self):
        rows = basic_values() + [value(2022, 12, 500)]
        result = compare_metrics(rows, [event()])[0]
        self.assertEqual(result.delta_abs, 8)
        self.assertEqual(result.delta_pct, 400)
        self.assertEqual(result.before['n_points'], 2)
        self.assertTrue(any('пересекает' in item for item in result.excluded_periods))
        boundary = compare_metrics(rows, [event(effective_date='2023-01-01')])[0]
        self.assertEqual(boundary.after['n_points'], 1)
        self.assertEqual(boundary.after['start'], '2023-02-01')

    def test_months_years_reconcile_without_summing_percentages(self):
        rows = [year_value(2021, 80, 'plan_completion')]
        rows += [value(2023, month, 60 if month <= 6 else 80, 'plan_completion') for month in range(1, 13)]
        result = compare_metrics(rows, [event(metric='plan_completion')])[0]
        self.assertEqual(result.after['value'], 70)
        self.assertEqual(result.delta_abs, -10)
        self.assertEqual(result.after['n_points'], 1)
        self.assertTrue(any('Проценты' in warning for warning in result.warnings))
        self.assertEqual(result.is_significant, 'insufficient_data')

    def test_incomplete_aggregated_year_is_not_used(self):
        rows = [year_value(2021, 10, 'audits_done')]
        rows += [value(2022, month, 100, 'audits_done') for month in range(1, 12)]
        rows += [value(2023, month, 2, 'audits_done') for month in range(1, 13)]
        result = compare_metrics(rows, [event(metric='audits_done')])[0]
        self.assertEqual(result.before['value'], 10)
        self.assertEqual(result.after['value'], 24)
        self.assertTrue(any('неполный' in excluded for excluded in result.excluded_periods))

    def test_native_total_and_months_are_not_double_counted_or_contradicted(self):
        rows = [year_value(2021, 12, 'audits_done'), year_value(2023, 24, 'audits_done')]
        rows += [value(2023, month, 2, 'audits_done') for month in range(1, 13)]
        result = compare_metrics(rows, [event(metric='audits_done')])[0]
        self.assertEqual(result.after['value'], 24)
        rows[1] = replace(rows[1], value=25)
        with self.assertRaises(ValueError):
            compare_metrics(rows, [event(metric='audits_done')])

    def test_repeated_reports_preserve_evidence_not_point_count(self):
        rows = basic_values()
        repeated = replace(rows[0], id='repeat-source', source=replace(rows[0].source, file='SYNTH_other.xlsx'))
        result = compare_metrics(rows + [repeated], [event()])[0]
        self.assertEqual(result.before['n_points'], 2)
        self.assertIn('repeat-source', result.metric_ids)
        with self.assertRaises(ValueError):
            compare_metrics(rows + [replace(repeated, value=123)], [event()])
        with self.assertRaises(ValueError):
            compare_metrics(rows + [replace(rows[0], metric='qa_score')], [event()])

    def test_nonfinite_unsourced_or_mixed_data_fail_closed(self):
        for bad in [replace(value(2022, 1, 2), value=float('nan')),
                    replace(value(2022, 1, 2), source=MetricSource('', '')),
                    replace(value(2022, 1, 2), is_synthetic=False)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_metrics([bad, *basic_values()[1:]], [event()])

    def test_fte_normalization_exact_scope_period_and_zero_safe(self):
        rows = basic_values('audits_done', before=4, after=12)
        fte = basic_values('fte_actual', before=2, after=4)
        result = compare_metrics(rows + fte, [event(metric='audits_done')])[0]
        self.assertEqual(result.normalized_delta, 1)
        self.assertEqual(len(result.normalization_sources), 4)
        fte[0] = replace(fte[0], value=0)
        self.assertIsNone(compare_metrics(rows + fte, [event(metric='audits_done')])[0].normalized_delta)
        other_scope = [replace(point, unit_scope='Другой отдел') for point in basic_values('fte_actual')]
        self.assertIsNone(compare_metrics(rows + other_scope, [event(metric='audits_done')])[0].normalized_delta)

    def test_did_requires_verified_control_and_exact_periods(self):
        rows = basic_values(before=10, after=15) + basic_values(scope='ДНМ', before=5, after=8)
        configuration = {'W1': {'scope': 'ДНМ', 'verified': True, 'reason': 'Проверен синтетический контроль'}}
        result = compare_metrics(rows, [event()], control_scopes=configuration)[0]
        self.assertEqual(result.method, 'diff_in_diff')
        self.assertEqual(result.adjusted_delta, 2)
        self.assertEqual(result.control_scope, 'ДНМ')
        self.assertEqual(compare_metrics(rows, [event()], control_scopes={'W1': 'ДНМ'})[0].method, 'simple')
        missing_period = [point for point in rows if not (point.unit_scope == 'ДНМ' and point.period_start == '2023-02-01')]
        self.assertEqual(compare_metrics(missing_period, [event()], control_scopes=configuration)[0].method, 'simple')
        affected_control = event('C1', 'ДНМ')
        self.assertNotEqual(compare_metrics(rows, [event(), affected_control], control_scopes=configuration)[0].method, 'diff_in_diff')

    def test_incompatible_control_does_not_erase_available_local_periods(self):
        rows = basic_values(n_before=3, n_after=3)
        rows += [year_value(2022, 2, scope='ДНМ'), year_value(2023, 2, scope='ДНМ')]
        result = compare_metrics(rows, [event()], control_scopes={'W1': {'scope': 'ДНМ', 'verified': True, 'reason': 'Контроль выбран'}})[0]
        self.assertEqual(result.method, 'trend_adjusted')
        self.assertEqual(result.before['n_points'], 3)
        self.assertEqual(result.after['n_points'], 3)

    def test_did_requires_same_known_currency_unit(self):
        configuration = {'W1': {'scope': 'ДНМ', 'verified': True, 'reason': 'Контроль выбран'}}
        for affected_unit, control_unit in [('KZT', 'USD'), ('KZT', ''), ('', 'KZT'), ('', ''), ('unknown', 'unknown')]:
            with self.subTest(affected_unit=affected_unit, control_unit=control_unit):
                rows = [replace(point, unit=affected_unit) for point in basic_values('budget_fact', before=10, after=15)]
                rows += [replace(point, unit=control_unit) for point in basic_values('budget_fact', scope='ДНМ', before=5, after=8)]
                result = compare_metrics(rows, [event(metric='budget_fact')], control_scopes=configuration)[0]
                self.assertEqual(result.method, 'simple')
                self.assertIsNone(result.control_scope)
                self.assertIsNone(result.adjusted_delta)
                self.assertTrue(any('единицы измерения' in warning for warning in result.warnings))
                self.assertTrue(all(not key.startswith('ДНМ:') for key in result.metric_ids))
        matching = [replace(point, unit='KZT') for point in basic_values('budget_fact', before=10, after=15)
                    + basic_values('budget_fact', scope='ДНМ', before=5, after=8)]
        result = compare_metrics(matching, [event(metric='budget_fact')], control_scopes=configuration)[0]
        self.assertEqual(result.method, 'diff_in_diff')
        self.assertEqual(result.adjusted_delta, 2)

    def test_incompatible_currency_does_not_coarsen_local_periods(self):
        rows = [replace(value(year, month, number, 'budget_fact'), unit='KZT')
                for year, number in [(2021, 10), (2023, 15)] for month in range(1, 13)]
        rows += [replace(year_value(year, number, 'budget_fact', 'ДНМ'), unit='USD')
                 for year, number in [(2021, 60), (2023, 96)]]
        result = compare_metrics(rows, [event(metric='budget_fact')],
                                 control_scopes={'W1': {'scope': 'ДНМ', 'verified': True, 'reason': 'Контроль выбран'}})[0]
        self.assertEqual(result.method, 'trend_adjusted')
        self.assertEqual(result.before['granularity'], 'month')
        self.assertEqual(result.before['n_points'], 12)
        self.assertEqual(result.after['n_points'], 12)
        self.assertEqual(result.delta_abs, 5)

    def test_trend_uses_elapsed_dates_not_observation_index(self):
        origin = date(2022, 1, 1).toordinal()
        rows = []
        for year, month in [(2022, 1), (2022, 3), (2022, 9), (2023, 1), (2023, 2)]:
            midpoint = (date(year, month, 1).toordinal() + date(year, month, calendar.monthrange(year, month)[1]).toordinal()) / 2
            rows.append(value(year, month, 10 + (midpoint - origin) * .01 + (5 if year == 2023 else 0)))
        result = compare_metrics(rows, [event()])[0]
        self.assertEqual(result.method, 'trend_adjusted')
        self.assertAlmostEqual(result.adjusted_delta, 5, places=8)


class AttributionAndHypothesisTests(unittest.TestCase):
    def test_confidence_precedence_and_mechanism_confirmation(self):
        e = event()
        cmp = compare_metrics(basic_values(), [e])[0]
        self.assertEqual(calculate_confidence(e, cmp, 'weakened')[0], 'low')
        self.assertEqual(calculate_confidence(e, cmp, 'refuted')[0], 'insufficient')
        self.assertEqual(calculate_confidence(replace(e, mechanism=''), cmp, 'upheld')[0], 'insufficient')
        self.assertEqual(calculate_confidence(replace(e, mechanism_confirmed=False), cmp, 'upheld')[0], 'low')
        strong = replace(cmp, method='diff_in_diff', warnings=[])
        self.assertEqual(calculate_confidence(e, strong, 'upheld')[0], 'high')
        self.assertEqual(calculate_confidence(e, strong, 'upheld', ['другое событие'])[0], 'medium')
        limited_simple = replace(cmp, warnings=['Предпосылка метода не подтверждена.'])
        self.assertEqual(calculate_confidence(e, limited_simple, 'upheld')[0], 'low')
        few = replace(cmp, before={**cmp.before, 'n_points': 1})
        self.assertEqual(calculate_confidence(e, few, 'upheld')[0], 'insufficient')

    def test_hiring_and_neighbor_events_are_alternatives(self):
        e = event('S1', 'ИТ', 'audits_done', type='unit_created')
        hire = event('HIRE1', 'ИТ', 'fte_actual', type='staff_increase', effective_date='2023-01-01', description='Найм сотрудников')
        rows = basic_values('audits_done', 'ИТ', 4, 10, 3, 3) + basic_values('fte_actual', 'ИТ', 4, 8, 3, 3)
        cmps = compare_metrics(rows, [e, hire])
        attribution = next(item for item in attribute_comparisons(rows, [e, hire], cmps) if item.event_id == 'S1')
        self.assertIn('HIRE1', attribution.evidence['alternative_events'])
        self.assertTrue(any('найм' in text.casefold() for text in attribution.alternative_explanations))
        self.assertEqual(attribution.opponent_outcome, 'weakened')
        text = json.dumps(asdict(attribution), ensure_ascii=False).casefold()
        self.assertFalse(re.search(r'привело к|вызвало|благодаря', text))

    def test_hypothesis_horizon_does_not_use_future_or_incomplete_coverage(self):
        e = event()
        rows = basic_values(n_before=3, n_after=3, before=2, after=2) + [value(2023, 4, 500)]
        cmp = compare_metrics(rows, [e])[0]
        hypothesis = Hypothesis('H1', 'Scenario', 'W1', 'Рост задержки', 'report_delay_days', 'up', '2023-03-31', 'ДККМ', 'W1', is_synthetic=True)
        self.assertEqual(check_hypotheses([hypothesis], [cmp], as_of='2023-03-30')[0].status, 'pending')
        self.assertEqual(check_hypotheses([hypothesis], [cmp], as_of='2023-04-30')[0].status, 'refuted')
        missing_january = [point for point in rows if point.period_start != '2023-01-01']
        incomplete = compare_metrics(missing_january, [e])[0]
        self.assertEqual(check_hypotheses([hypothesis], [incomplete], as_of='2023-04-30')[0].status, 'inconclusive')
        self.assertEqual(hypothesis.status, 'pending')
        assessed = [replace(hypothesis, status=status) for status in ('confirmed', 'refuted', 'pending', 'inconclusive')]
        self.assertEqual(hypothesis_accuracy(assessed)['accuracy'], .5)

    def test_recommendations_are_adverse_only_and_quantities_require_diverse_cases(self):
        e = event()
        rows = basic_values()
        cmp = compare_metrics(rows, [e])[0]
        attrs = attribute_comparisons(rows, [e], [cmp])
        pattern = f"{e.type}: {' '.join(e.mechanism.casefold().split())}"
        record = KnowledgeRecord('K', pattern, {'cases': [dict(event_date=f'{year}-12-23', unit='ДККМ', delta_pct=delta) for year, delta in [(2019, 10), (2020, 20), (2021, 30)]]}, {'metric': 'report_delay_days'}, 'medium', ['a', 'b', 'c'], 3, True, 3)
        rec = recommend_actions([e], attrs, [cmp], [record])[0]
        self.assertEqual(rec.expected_effect['historical_observed_pct_range'], [10, 30])
        self.assertIsNone(rec.redline_id)
        self.assertEqual(rec.target_units, ['ДККМ'])
        one = replace(record, n_cases=1, diversity=1)
        self.assertNotIn('historical_observed_pct_range', recommend_actions([e], attrs, [cmp], [one])[0].expected_effect)
        improved = replace(cmp, delta_abs=-1)
        self.assertEqual(recommend_actions([e], attrs, [improved], []), [])
        context_metric = replace(cmp, metric='findings_total')
        self.assertEqual(recommend_actions([e], attrs, [context_metric], []), [])

    def test_core_overlap_without_numbers_yields_only_cautious_document_review(self):
        e = event('D1', type='func_overlap')
        rec = recommend_actions([e], [], [], [])[0]
        self.assertIsNone(rec.expected_effect['metric'])
        self.assertEqual(rec.confidence, 'low')
        self.assertEqual(rec.target_units, e.units)
        self.assertTrue(rec.requires_human_decision)

    def test_synthetic_workflow_detects_w1_s1_and_hiring(self):
        from scripts.gen_synthetic_reports import generate_reports
        from qaitu.metrics_ingest import ingest_reports
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_reports(Path(directory))
            ingestion = ingest_reports([(Path(path).name, Path(path).read_bytes()) for path in manifest['files']], is_synthetic=True)
        events = [StructuralEvent(**item) for item in manifest['events']]
        cmps = compare_metrics(ingestion.values, events, control_scopes=manifest['control_scopes'])
        attrs = attribute_comparisons(ingestion.values, events, cmps)
        w1 = next(item for item in attrs if item.event_id == 'W1')
        s1 = next(item for item in attrs if item.event_id == 'S1')
        self.assertIn(w1.confidence, {'medium', 'high'})
        self.assertTrue(any('HIRE1' in text for text in s1.alternative_explanations))
        self.assertTrue(all(item.evidence['metrics'] for item in attrs))


if __name__ == '__main__':
    unittest.main()
