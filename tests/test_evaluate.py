"""Tests for evaluate.py — fully testable without numpy/sklearn."""

import sys
import unittest

sys.path.insert(0, "/workspace")

from lg_cotrain.evaluate import compute_metrics, evaluate_pseudo_labels, _compute_f1_pure


class TestComputeMetricsPerfect(unittest.TestCase):
    """Perfect predictions."""

    def test_error_rate_zero(self):
        m = compute_metrics([0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertAlmostEqual(m["error_rate"], 0.0)

    def test_macro_f1_one(self):
        m = compute_metrics([0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertAlmostEqual(m["macro_f1"], 1.0)

    def test_per_class_f1_all_ones(self):
        m = compute_metrics([0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3])
        for f1 in m["per_class_f1"]:
            self.assertAlmostEqual(f1, 1.0)


class TestComputeMetricsAllWrong(unittest.TestCase):
    """All predictions wrong."""

    def test_error_rate_100(self):
        m = compute_metrics([0, 0, 0, 0], [1, 1, 1, 1])
        self.assertAlmostEqual(m["error_rate"], 100.0)

    def test_macro_f1_zero(self):
        m = compute_metrics([0, 0, 0, 0], [1, 1, 1, 1])
        self.assertAlmostEqual(m["macro_f1"], 0.0)


class TestComputeMetricsPartial(unittest.TestCase):
    """Partial correctness."""

    def test_error_rate_between_0_and_100(self):
        m = compute_metrics([0, 1, 0, 1], [0, 0, 0, 1])
        self.assertGreater(m["error_rate"], 0.0)
        self.assertLess(m["error_rate"], 100.0)

    def test_macro_f1_between_0_and_1(self):
        m = compute_metrics([0, 1, 0, 1], [0, 0, 0, 1])
        self.assertGreater(m["macro_f1"], 0.0)
        self.assertLess(m["macro_f1"], 1.0)

    def test_per_class_f1_is_list(self):
        m = compute_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 1])
        self.assertIsInstance(m["per_class_f1"], list)
        self.assertEqual(len(m["per_class_f1"]), 3)

    def test_known_error_rate(self):
        """3 out of 4 correct → error_rate = 25%."""
        m = compute_metrics([0, 1, 2, 3], [0, 1, 2, 0])
        self.assertAlmostEqual(m["error_rate"], 25.0)


class TestComputeF1Pure(unittest.TestCase):
    """Test pure-Python F1 computation directly."""

    def test_perfect(self):
        macro, per = _compute_f1_pure([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertAlmostEqual(macro, 1.0)
        self.assertEqual(len(per), 2)

    def test_all_wrong(self):
        macro, per = _compute_f1_pure([0, 0, 0], [1, 1, 1])
        self.assertAlmostEqual(macro, 0.0)

    def test_known_binary_case(self):
        """tp=2, fp=1, fn=0 for class 1; tp=1, fp=0, fn=1 for class 0."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 0]
        macro, per = _compute_f1_pure(y_true, y_pred)
        # class 0: tp=1, fp=0, fn=1 → p=1, r=0.5, f1=2/3
        # class 1: tp=2, fp=1, fn=0 → p=2/3, r=1, f1=4/5=0.8
        self.assertAlmostEqual(per[0], 2/3, places=4)
        self.assertAlmostEqual(per[1], 0.8, places=4)
        self.assertAlmostEqual(macro, (2/3 + 0.8) / 2, places=4)


class TestEnsembleAveraging(unittest.TestCase):
    """Test the ensemble averaging logic conceptually."""

    def test_average_two_distributions(self):
        """model1=[0.8,0.2], model2=[0.4,0.6] → avg=[0.6,0.4] → pred=0"""
        probs1 = [0.8, 0.2]
        probs2 = [0.4, 0.6]
        avg = [(p1 + p2) / 2 for p1, p2 in zip(probs1, probs2)]
        self.assertAlmostEqual(avg[0], 0.6)
        self.assertAlmostEqual(avg[1], 0.4)
        self.assertEqual(avg.index(max(avg)), 0)

    def test_average_ties_to_first(self):
        """If both models agree, average still agrees."""
        probs1 = [0.9, 0.1]
        probs2 = [0.7, 0.3]
        avg = [(p1 + p2) / 2 for p1, p2 in zip(probs1, probs2)]
        self.assertEqual(avg.index(max(avg)), 0)

    def test_average_can_flip(self):
        """Second model can override first if stronger."""
        probs1 = [0.6, 0.4]
        probs2 = [0.2, 0.8]
        avg = [(p1 + p2) / 2 for p1, p2 in zip(probs1, probs2)]
        self.assertEqual(avg.index(max(avg)), 1)


class TestEvaluatePseudoLabels(unittest.TestCase):
    """Test pseudo-label accuracy computation."""

    def test_all_correct(self):
        acc = evaluate_pseudo_labels(["a", "b", "c"], ["a", "b", "c"])
        self.assertAlmostEqual(acc, 100.0)

    def test_all_wrong(self):
        acc = evaluate_pseudo_labels(["a", "b", "c"], ["x", "y", "z"])
        self.assertAlmostEqual(acc, 0.0)

    def test_partial(self):
        acc = evaluate_pseudo_labels(
            ["a", "b", "a", "b", "a"],
            ["a", "b", "a", "a", "a"],
        )
        self.assertAlmostEqual(acc, 80.0)

    def test_single_sample(self):
        acc = evaluate_pseudo_labels(["a"], ["a"])
        self.assertAlmostEqual(acc, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
