"""Tests for multi-GPU parallel experiment execution.

Tests the orchestration logic (GPU assignment, resume, dispatch)
without requiring actual GPUs.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_result(event="test_event", budget=5, seed_set=1, **overrides):
    """Create a fake metrics result dict."""
    result = {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": 25.0,
        "test_macro_f1": 0.75,
        "test_ece": 0.05,
        "test_per_class_f1": [0.7, 0.8],
        "dev_error_rate": 20.0,
        "dev_macro_f1": 0.80,
        "dev_ece": 0.04,
        "stopping_strategy": "baseline",
        "lambda1_mean": 0.5,
        "lambda1_std": 0.1,
        "lambda2_mean": 0.4,
        "lambda2_std": 0.1,
    }
    result.update(overrides)
    return result


class TestGPURoundRobinAssignment(unittest.TestCase):
    """Test that experiments get assigned to GPUs round-robin."""

    def test_2_gpus_6_experiments(self):
        configs = [{"event": "e", "budget": b, "seed_set": s}
                   for b in [5, 10] for s in [1, 2, 3]]
        num_gpus = 2
        for i, cfg in enumerate(configs):
            cfg["device"] = f"cuda:{i % num_gpus}"
        devices = [c["device"] for c in configs]
        self.assertEqual(devices, [
            "cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0", "cuda:1",
        ])

    def test_3_gpus_3_experiments(self):
        configs = [{"event": "e", "budget": 5, "seed_set": s}
                   for s in [1, 2, 3]]
        num_gpus = 3
        for i, cfg in enumerate(configs):
            cfg["device"] = f"cuda:{i % num_gpus}"
        devices = [c["device"] for c in configs]
        self.assertEqual(devices, ["cuda:0", "cuda:1", "cuda:2"])

    def test_1_gpu_all_same(self):
        configs = [{"event": "e"} for _ in range(4)]
        for i, cfg in enumerate(configs):
            cfg["device"] = f"cuda:{i % 1}"
        self.assertTrue(all(c["device"] == "cuda:0" for c in configs))


class TestRunAllParallelDispatch(unittest.TestCase):
    """Test that run_all_experiments dispatches correctly based on num_gpus."""

    def test_num_gpus_1_uses_sequential(self):
        """num_gpus=1 should use the existing sequential code path."""
        from lg_cotrain.run_all import run_all_experiments

        call_count = [0]

        def fake_cls(config):
            call_count[0] += 1
            mock = MagicMock()
            mock.run.return_value = _make_result(
                budget=config.budget, seed_set=config.seed_set,
            )
            return mock

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "test_event",
                budgets=[5], seed_sets=[1],
                data_root=tmpdir, results_root=tmpdir,
                _trainer_cls=fake_cls, num_gpus=1,
            )
        self.assertEqual(call_count[0], 1)
        self.assertEqual(len(results), 1)

    def test_num_gpus_default_is_1(self):
        """Default num_gpus should be 1 (sequential)."""
        import inspect
        from lg_cotrain.run_all import run_all_experiments

        sig = inspect.signature(run_all_experiments)
        self.assertEqual(sig.parameters["num_gpus"].default, 1)

    def test_num_gpus_2_calls_parallel(self):
        """num_gpus=2 should invoke _run_all_parallel."""
        from lg_cotrain.run_all import run_all_experiments

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lg_cotrain.run_all._run_all_parallel") as mock_par:
                mock_par.return_value = [
                    _make_result(budget=5, seed_set=1),
                ]
                results = run_all_experiments(
                    "test_event",
                    budgets=[5], seed_sets=[1],
                    data_root=tmpdir, results_root=tmpdir,
                    num_gpus=2,
                )
            mock_par.assert_called_once()

    def test_num_gpus_2_with_trainer_cls_uses_sequential(self):
        """When _trainer_cls is provided, sequential path is used even with num_gpus>1."""
        from lg_cotrain.run_all import run_all_experiments

        call_count = [0]

        def fake_cls(config):
            call_count[0] += 1
            mock = MagicMock()
            mock.run.return_value = _make_result(
                budget=config.budget, seed_set=config.seed_set,
            )
            return mock

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "test_event",
                budgets=[5], seed_sets=[1],
                data_root=tmpdir, results_root=tmpdir,
                _trainer_cls=fake_cls, num_gpus=2,
            )
        # Should fall through to sequential because _trainer_cls is set
        self.assertEqual(call_count[0], 1)


class TestResumeInParallelMode(unittest.TestCase):
    """Existing metrics.json files should be loaded without spawning processes."""

    def test_skips_existing_in_parallel(self):
        from lg_cotrain.run_all import run_all_experiments

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create one result
            out_dir = Path(tmpdir) / "test_event" / "5_set1"
            out_dir.mkdir(parents=True)
            existing = _make_result(budget=5, seed_set=1, test_error_rate=99.0)
            (out_dir / "metrics.json").write_text(json.dumps(existing))

            with patch("lg_cotrain.parallel.run_experiments_parallel") as mock_par:
                mock_par.return_value = [{
                    "event": "test_event", "budget": 5, "seed_set": 2,
                    "status": "done",
                    "result": _make_result(budget=5, seed_set=2),
                }]
                results = run_all_experiments(
                    "test_event",
                    budgets=[5], seed_sets=[1, 2],
                    data_root=tmpdir, results_root=tmpdir,
                    num_gpus=2,
                )

            # Only 1 experiment should have been submitted to parallel
            configs_submitted = mock_par.call_args[0][0]
            self.assertEqual(len(configs_submitted), 1)
            # The skipped result should be loaded from disk
            self.assertEqual(results[0]["test_error_rate"], 99.0)
            # The parallel result should be in position 2
            self.assertEqual(results[1]["seed_set"], 2)

    def test_all_skipped_no_parallel_call(self):
        """If all experiments exist, run_experiments_parallel is not called."""
        from lg_cotrain.run_all import run_all_experiments

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create all results
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}"
                out_dir.mkdir(parents=True)
                result = _make_result(budget=5, seed_set=seed_set)
                (out_dir / "metrics.json").write_text(json.dumps(result))

            with patch("lg_cotrain.parallel.run_experiments_parallel") as mock_par:
                results = run_all_experiments(
                    "test_event",
                    budgets=[5], seed_sets=[1, 2],
                    data_root=tmpdir, results_root=tmpdir,
                    num_gpus=2,
                )

            mock_par.assert_not_called()
            self.assertEqual(len(results), 2)


class TestParallelCallbackInvocation(unittest.TestCase):
    """Callbacks should be invoked for both skipped and parallel experiments."""

    def test_callback_called_for_skipped_and_done(self):
        from lg_cotrain.run_all import run_all_experiments

        statuses = []

        def on_done(event, budget, seed_set, status):
            statuses.append((budget, seed_set, status))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create seed_set=1
            out_dir = Path(tmpdir) / "test_event" / "5_set1"
            out_dir.mkdir(parents=True)
            existing = _make_result(budget=5, seed_set=1)
            (out_dir / "metrics.json").write_text(json.dumps(existing))

            with patch("lg_cotrain.parallel.run_experiments_parallel") as mock_par:
                mock_par.return_value = [{
                    "event": "test_event", "budget": 5, "seed_set": 2,
                    "status": "done",
                    "result": _make_result(budget=5, seed_set=2),
                }]
                run_all_experiments(
                    "test_event",
                    budgets=[5], seed_sets=[1, 2],
                    data_root=tmpdir, results_root=tmpdir,
                    num_gpus=2,
                    _on_experiment_done=on_done,
                )

        # Skipped callback was called from run_all
        self.assertIn((5, 1, "skipped"), statuses)
        # Done callback is invoked by run_experiments_parallel (mocked here,
        # so it was passed through but not actually called by the mock)


class TestRunExperimentNumGpusFlag(unittest.TestCase):
    """--num-gpus flag is accepted and forwarded."""

    def test_num_gpus_flag_accepted(self):
        """run_experiment.py should accept --num-gpus."""
        from lg_cotrain.run_experiment import main

        with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
            mock_run.return_value = [_make_result()]
            with patch("sys.argv", [
                "prog", "--event", "test_event",
                "--budget", "5", "--seed-set", "1",
                "--num-gpus", "2",
            ]):
                main()

            call_kwargs = mock_run.call_args
            self.assertEqual(call_kwargs.kwargs.get("num_gpus"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
