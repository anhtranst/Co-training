"""Tests for dashboard.py — HTML results dashboard generator.

Pure-Python tests: no torch/numpy/transformers required.
"""

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/workspace")

from lg_cotrain.dashboard import (
    EVENTS,
    build_lambda_pivot,
    build_overall_means,
    build_pivot_data,
    collect_all_metrics,
    compute_summary_cards,
    count_expected_experiments,
    format_event_name,
    generate_html,
    get_event_class_count,
)
from lg_cotrain.run_all import BUDGETS, SEED_SETS


def _make_metric(event="california_wildfires_2018", budget=5, seed_set=1,
                 macro_f1=0.55, error_rate=35.0, num_classes=10):
    """Build a metrics dict matching trainer.run() output."""
    return {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": error_rate,
        "test_macro_f1": macro_f1,
        "test_per_class_f1": [macro_f1] * num_classes,
        "dev_error_rate": error_rate - 1.0,
        "dev_macro_f1": macro_f1 + 0.01,
        "lambda1_mean": 1.08,
        "lambda1_std": 0.21,
        "lambda2_mean": 0.56,
        "lambda2_std": 0.17,
    }


def _write_metric(tmpdir, metric):
    """Write a metric dict to the proper path under tmpdir."""
    path = (Path(tmpdir) / metric["event"]
            / f"{metric['budget']}_set{metric['seed_set']}" / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metric))


class TestFormatEventName(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            format_event_name("california_wildfires_2018"),
            "California Wildfires 2018",
        )

    def test_multi_word(self):
        self.assertEqual(
            format_event_name("hurricane_harvey_2017"),
            "Hurricane Harvey 2017",
        )


class TestCollectAllMetrics(unittest.TestCase):
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_all_metrics(tmpdir)
        self.assertEqual(result, [])

    def test_single_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = _make_metric()
            _write_metric(tmpdir, m)
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event"], "california_wildfires_2018")

    def test_multiple_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for s in SEED_SETS:
                _write_metric(tmpdir, _make_metric(seed_set=s))
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 3)

    def test_skips_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a valid one
            _write_metric(tmpdir, _make_metric(seed_set=1))
            # Write a malformed one
            bad_path = (Path(tmpdir) / "california_wildfires_2018"
                        / "5_set2" / "metrics.json")
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_path.write_text("{invalid json")
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)

    def test_returns_dicts_with_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(tmpdir, _make_metric())
            result = collect_all_metrics(tmpdir)
        for key in ["event", "budget", "seed_set", "test_macro_f1"]:
            self.assertIn(key, result[0])


class TestCountExpectedExperiments(unittest.TestCase):
    def test_count_is_120(self):
        self.assertEqual(count_expected_experiments(), 120)


class TestGetEventClassCount(unittest.TestCase):
    def test_from_metrics(self):
        metrics = [_make_metric(num_classes=8)]
        counts = get_event_class_count(metrics)
        self.assertEqual(counts["california_wildfires_2018"], 8)

    def test_missing_event_defaults_to_10(self):
        counts = get_event_class_count([])
        for event in EVENTS:
            self.assertEqual(counts[event], 10)

    def test_mixed_class_counts(self):
        metrics = [
            _make_metric(event="california_wildfires_2018", num_classes=10),
            _make_metric(event="canada_wildfires_2016", num_classes=8),
        ]
        counts = get_event_class_count(metrics)
        self.assertEqual(counts["california_wildfires_2018"], 10)
        self.assertEqual(counts["canada_wildfires_2016"], 8)


class TestBuildPivotData(unittest.TestCase):
    def test_empty_metrics(self):
        pivot = build_pivot_data([])
        for event in EVENTS:
            for budget in BUDGETS:
                self.assertIsNone(pivot[event][budget]["f1_mean"])
                self.assertEqual(pivot[event][budget]["count"], 0)

    def test_single_seed_no_std(self):
        metrics = [_make_metric(budget=5, seed_set=1, macro_f1=0.60)]
        pivot = build_pivot_data(metrics)
        entry = pivot["california_wildfires_2018"][5]
        self.assertAlmostEqual(entry["f1_mean"], 0.60)
        self.assertIsNone(entry["f1_std"])
        self.assertEqual(entry["count"], 1)

    def test_three_seeds_mean_std(self):
        f1_vals = [0.50, 0.55, 0.60]
        metrics = [
            _make_metric(budget=10, seed_set=s, macro_f1=f)
            for s, f in zip(SEED_SETS, f1_vals)
        ]
        pivot = build_pivot_data(metrics)
        entry = pivot["california_wildfires_2018"][10]
        self.assertAlmostEqual(entry["f1_mean"], statistics.mean(f1_vals))
        self.assertAlmostEqual(entry["f1_std"], statistics.stdev(f1_vals))
        self.assertEqual(entry["count"], 3)

    def test_multiple_events(self):
        metrics = [
            _make_metric(event="california_wildfires_2018", budget=5, macro_f1=0.60),
            _make_metric(event="canada_wildfires_2016", budget=5, macro_f1=0.50),
        ]
        pivot = build_pivot_data(metrics)
        self.assertAlmostEqual(pivot["california_wildfires_2018"][5]["f1_mean"], 0.60)
        self.assertAlmostEqual(pivot["canada_wildfires_2016"][5]["f1_mean"], 0.50)


class TestBuildOverallMeans(unittest.TestCase):
    def test_single_event(self):
        metrics = [_make_metric(budget=5, macro_f1=0.60, error_rate=30.0)]
        pivot = build_pivot_data(metrics)
        overall = build_overall_means(pivot)
        self.assertAlmostEqual(overall[5]["f1_mean"], 0.60)
        self.assertAlmostEqual(overall[5]["err_mean"], 30.0)

    def test_multiple_events_averaged(self):
        metrics = [
            _make_metric(event="california_wildfires_2018", budget=5, macro_f1=0.60, error_rate=30.0),
            _make_metric(event="canada_wildfires_2016", budget=5, macro_f1=0.40, error_rate=50.0),
        ]
        pivot = build_pivot_data(metrics)
        overall = build_overall_means(pivot)
        self.assertAlmostEqual(overall[5]["f1_mean"], 0.50)
        self.assertAlmostEqual(overall[5]["err_mean"], 40.0)

    def test_missing_budget_is_none(self):
        pivot = build_pivot_data([])
        overall = build_overall_means(pivot)
        for budget in BUDGETS:
            self.assertIsNone(overall[budget]["f1_mean"])


class TestBuildLambdaPivot(unittest.TestCase):
    def test_lambda_values_extracted(self):
        metrics = [_make_metric()]
        pivot = build_lambda_pivot(metrics)
        entry = pivot["california_wildfires_2018"][5]
        self.assertAlmostEqual(entry["l1_mean"], 1.08)
        self.assertAlmostEqual(entry["l2_mean"], 0.56)

    def test_averaged_across_seeds(self):
        metrics = [
            _make_metric(seed_set=1),
            _make_metric(seed_set=2),
            _make_metric(seed_set=3),
        ]
        # All have lambda1_mean=1.08, so average should be 1.08
        pivot = build_lambda_pivot(metrics)
        self.assertAlmostEqual(pivot["california_wildfires_2018"][5]["l1_mean"], 1.08)

    def test_missing_is_none(self):
        pivot = build_lambda_pivot([])
        self.assertIsNone(pivot["california_wildfires_2018"][5]["l1_mean"])


class TestComputeSummaryCards(unittest.TestCase):
    def test_experiment_count(self):
        metrics = [_make_metric()]
        s = compute_summary_cards(metrics)
        self.assertEqual(s["completed"], 1)
        self.assertEqual(s["total"], 120)

    def test_percentage(self):
        metrics = [_make_metric() for _ in range(12)]
        s = compute_summary_cards(metrics)
        self.assertAlmostEqual(s["pct"], 10.0)

    def test_avg_f1(self):
        metrics = [
            _make_metric(macro_f1=0.50),
            _make_metric(macro_f1=0.60, seed_set=2),
        ]
        s = compute_summary_cards(metrics)
        self.assertAlmostEqual(s["avg_f1"], 0.55)

    def test_avg_error_rate(self):
        metrics = [
            _make_metric(error_rate=30.0),
            _make_metric(error_rate=40.0, seed_set=2),
        ]
        s = compute_summary_cards(metrics)
        self.assertAlmostEqual(s["avg_err"], 35.0)

    def test_disaster_count(self):
        metrics = [
            _make_metric(event="california_wildfires_2018"),
            _make_metric(event="canada_wildfires_2016"),
        ]
        s = compute_summary_cards(metrics)
        self.assertEqual(s["disasters_done"], 2)

    def test_empty_metrics(self):
        s = compute_summary_cards([])
        self.assertEqual(s["completed"], 0)
        self.assertIsNone(s["avg_f1"])
        self.assertIsNone(s["avg_err"])
        self.assertEqual(s["disasters_done"], 0)


class TestGenerateHtml(unittest.TestCase):
    def test_returns_string(self):
        html = generate_html([], "/tmp/fake")
        self.assertIsInstance(html, str)

    def test_contains_doctype(self):
        html = generate_html([], "/tmp/fake")
        self.assertTrue(html.strip().startswith("<!DOCTYPE html>"))

    def test_contains_title(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("LG-CoTrain", html)

    def test_contains_summary_cards(self):
        metrics = [_make_metric(macro_f1=0.616, error_rate=28.82)]
        html = generate_html(metrics, "/tmp/fake")
        self.assertIn("1 / 120", html)
        self.assertIn("0.616", html)

    def test_contains_pivot_tables(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("Macro-F1 by Disaster", html)
        self.assertIn("Error Rate", html)
        self.assertIn("Lambda Weights", html)

    def test_contains_all_results_div(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn('id="all-view"', html)
        self.assertIn("All Experiment Results", html)

    def test_contains_toggle_buttons(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("Pivot Summary", html)
        self.assertIn("All Results", html)
        self.assertIn("showView", html)

    def test_partial_results_no_error(self):
        metrics = [_make_metric()]
        html = generate_html(metrics, "/tmp/fake")
        self.assertIn("California Wildfires 2018", html)
        self.assertIn("---", html)  # Missing cells

    def test_empty_results_no_error(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("0 / 120", html)
        self.assertIn("N/A", html)

    def test_color_classes_present(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("cell-pending", html)

    def test_event_names_formatted(self):
        metrics = [_make_metric(event="hurricane_harvey_2017")]
        html = generate_html(metrics, "/tmp/fake")
        self.assertIn("Hurricane Harvey 2017", html)


class TestDashboardCLI(unittest.TestCase):
    def test_default_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir]):
                from lg_cotrain.dashboard import main
                main()
            self.assertTrue((Path(tmpdir) / "dashboard.html").exists())

    def test_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "custom.html")
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir,
                                     "--output", output]):
                from lg_cotrain.dashboard import main
                main()
            self.assertTrue(Path(output).exists())

    def test_writes_valid_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(tmpdir, _make_metric())
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir]):
                from lg_cotrain.dashboard import main
                main()
            content = (Path(tmpdir) / "dashboard.html").read_text()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("California Wildfires 2018", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
