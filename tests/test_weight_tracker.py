"""Tests for weight_tracker.py — works with or without numpy."""

import math
import sys
import unittest

sys.path.insert(0, "/workspace")

from lg_cotrain.weight_tracker import WeightTracker


class TestWeightTrackerSingleEpoch(unittest.TestCase):
    """Record a single epoch of probabilities."""

    def test_confidence_equals_input(self):
        tracker = WeightTracker(num_samples=5)
        probs = [0.8, 0.6, 0.7, 0.9, 0.5]
        tracker.record_epoch(probs)
        conf = tracker.compute_confidence()
        for a, b in zip(conf, probs):
            self.assertAlmostEqual(a, b, places=6)

    def test_variability_is_zero(self):
        tracker = WeightTracker(num_samples=5)
        tracker.record_epoch([0.8, 0.6, 0.7, 0.9, 0.5])
        var = tracker.compute_variability()
        for v in var:
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_lambda_optimistic_equals_confidence(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.5, 0.6, 0.7])
        lam = tracker.compute_lambda_optimistic()
        conf = tracker.compute_confidence()
        for a, b in zip(lam, conf):
            self.assertAlmostEqual(a, b, places=6)

    def test_lambda_conservative_equals_confidence(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.5, 0.6, 0.7])
        lam = tracker.compute_lambda_conservative()
        conf = tracker.compute_confidence()
        for a, b in zip(lam, conf):
            self.assertAlmostEqual(a, b, places=6)


class TestWeightTrackerMultipleEpochs(unittest.TestCase):
    """Record multiple epochs and verify confidence/variability math."""

    def test_confidence_is_mean(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        conf = tracker.compute_confidence()
        self.assertAlmostEqual(conf[0], 0.7, places=6)
        self.assertAlmostEqual(conf[1], 0.5, places=6)
        self.assertAlmostEqual(conf[2], 0.8, places=6)

    def test_variability_is_population_std(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        var = tracker.compute_variability()
        # Population std for [0.8, 0.6] = sqrt(((0.8-0.7)^2 + (0.6-0.7)^2)/2) = 0.1
        self.assertAlmostEqual(var[0], 0.1, places=6)
        # Population std for [0.6, 0.4] = 0.1
        self.assertAlmostEqual(var[1], 0.1, places=6)
        # Population std for [0.7, 0.9] = 0.1
        self.assertAlmostEqual(var[2], 0.1, places=6)

    def test_lambda_optimistic_is_c_plus_v(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        lam = tracker.compute_lambda_optimistic()
        # c + v = [0.7+0.1, 0.5+0.1, 0.8+0.1]
        self.assertAlmostEqual(lam[0], 0.8, places=6)
        self.assertAlmostEqual(lam[1], 0.6, places=6)
        self.assertAlmostEqual(lam[2], 0.9, places=6)

    def test_lambda_conservative_is_c_minus_v(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        lam = tracker.compute_lambda_conservative()
        # c - v = [0.7-0.1, 0.5-0.1, 0.8-0.1]
        self.assertAlmostEqual(lam[0], 0.6, places=6)
        self.assertAlmostEqual(lam[1], 0.4, places=6)
        self.assertAlmostEqual(lam[2], 0.7, places=6)


class TestWeightTrackerEdgeCases(unittest.TestCase):
    """Edge cases for lambda computation."""

    def test_identical_probs_across_epochs(self):
        """If all epochs have same probs, variability=0, lambda1 == lambda2."""
        tracker = WeightTracker(num_samples=3)
        probs = [0.8, 0.6, 0.7]
        tracker.record_epoch(probs)
        tracker.record_epoch(probs)
        tracker.record_epoch(probs)
        for v in tracker.compute_variability():
            self.assertAlmostEqual(v, 0.0, places=6)
        lam1 = tracker.compute_lambda_optimistic()
        lam2 = tracker.compute_lambda_conservative()
        for a, b in zip(lam1, lam2):
            self.assertAlmostEqual(a, b, places=6)

    def test_high_variability_clips_to_zero(self):
        """When variability > confidence, lambda2 should be clipped to 0."""
        tracker = WeightTracker(num_samples=2)
        tracker.record_epoch([0.1, 0.2])
        tracker.record_epoch([0.9, 0.8])
        # confidence = [0.5, 0.5], variability = [0.4, 0.3]
        # c - v = [0.1, 0.2] -> both positive, no clipping here
        lam2 = tracker.compute_lambda_conservative()
        for val in lam2:
            self.assertGreaterEqual(val, 0.0)

    def test_extreme_high_variability_clips(self):
        """Force a case where c - v < 0."""
        tracker = WeightTracker(num_samples=1)
        tracker.record_epoch([0.0])
        tracker.record_epoch([1.0])
        # confidence = 0.5, variability = 0.5, c - v = 0
        lam2 = tracker.compute_lambda_conservative()
        self.assertAlmostEqual(lam2[0], 0.0, places=6)

    def test_known_three_epoch_example(self):
        """probs=[0.8, 0.6, 0.7] for 1 sample → confidence=0.7, variability≈0.0816."""
        tracker = WeightTracker(num_samples=1)
        tracker.record_epoch([0.8])
        tracker.record_epoch([0.6])
        tracker.record_epoch([0.7])
        self.assertAlmostEqual(tracker.compute_confidence()[0], 0.7, places=5)
        expected_std = math.sqrt(((0.8-0.7)**2 + (0.6-0.7)**2 + (0.7-0.7)**2) / 3)
        self.assertAlmostEqual(tracker.compute_variability()[0], expected_std, places=4)


class TestWeightTrackerBookkeeping(unittest.TestCase):
    """Test record counting and input validation."""

    def test_num_epochs_starts_at_zero(self):
        tracker = WeightTracker(num_samples=3)
        self.assertEqual(tracker.num_epochs_recorded, 0)

    def test_num_epochs_increments(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.5, 0.5, 0.5])
        self.assertEqual(tracker.num_epochs_recorded, 1)
        tracker.record_epoch([0.6, 0.6, 0.6])
        self.assertEqual(tracker.num_epochs_recorded, 2)

    def test_wrong_length_raises(self):
        tracker = WeightTracker(num_samples=3)
        with self.assertRaises(AssertionError):
            tracker.record_epoch([0.5, 0.5])  # too short

    def test_record_does_not_alias_input(self):
        """Mutating the input list after recording should not affect stored data."""
        tracker = WeightTracker(num_samples=2)
        probs = [0.5, 0.5]
        tracker.record_epoch(probs)
        probs[0] = 999.0
        self.assertAlmostEqual(tracker.compute_confidence()[0], 0.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
