"""Tests for per-experiment Optuna hyperparameter tuning.

Pure-Python tests that mock optuna and the trainer. No GPU required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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


class MockTrial:
    """Mock Optuna trial that records suggest calls."""

    def __init__(self):
        self.suggestions = {}
        self.number = 0

    def suggest_float(self, name, low, high, log=False):
        val = (low + high) / 2 if not log else (low * high) ** 0.5
        self.suggestions[name] = val
        return val

    def suggest_categorical(self, name, choices):
        val = choices[0]
        self.suggestions[name] = val
        return val

    def suggest_int(self, name, low, high):
        val = (low + high) // 2
        self.suggestions[name] = val
        return val


class TestSearchSpace(unittest.TestCase):
    """Verify objective samples 6 hyperparameters from a trial."""

    def test_objective_samples_6_params(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result()

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        objective(trial)

        expected_params = {"lr", "batch_size", "cotrain_epochs",
                           "finetune_patience", "weight_decay", "warmup_ratio"}
        self.assertEqual(set(trial.suggestions.keys()), expected_params)

    def test_lr_is_log_scale(self):
        """LR should be sampled with log=True."""
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result()

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        # Use a real-ish mock trial that tracks log param
        log_params = []
        trial = MockTrial()
        original_suggest_float = trial.suggest_float

        def tracking_suggest_float(name, low, high, log=False):
            if log:
                log_params.append(name)
            return original_suggest_float(name, low, high, log=log)

        trial.suggest_float = tracking_suggest_float
        objective(trial)

        self.assertIn("lr", log_params)


class TestObjectiveReturnValue(unittest.TestCase):
    """Verify objective returns dev_macro_f1."""

    def test_returns_dev_macro_f1(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(
            dev_macro_f1=0.8765,
        )

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        result = objective(trial)

        self.assertEqual(result, 0.8765)


class TestObjectiveUsesDevNotTest(unittest.TestCase):
    """Verify the objective uses dev_macro_f1, not test_macro_f1."""

    def test_uses_dev_not_test(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(
            dev_macro_f1=0.80,
            test_macro_f1=0.90,  # different from dev
        )

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        result = objective(trial)

        # Should return dev, not test
        self.assertEqual(result, 0.80)
        self.assertNotEqual(result, 0.90)


class TestBestParamsSaved(unittest.TestCase):
    """Verify best_params.json is written after study completes."""

    def test_saves_best_params_json(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

        mock_trainer = MagicMock()
        call_count = [0]

        def fake_run():
            call_count[0] += 1
            return _make_result(dev_macro_f1=0.70 + call_count[0] * 0.01)

        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=3, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            # Check file was created
            best_path = Path(tmpdir) / "test_event" / "5_set1" / "best_params.json"
            self.assertTrue(best_path.exists())

            # Check contents
            with open(best_path) as f:
                saved = json.load(f)

            self.assertEqual(saved["event"], "test_event")
            self.assertEqual(saved["budget"], 5)
            self.assertEqual(saved["seed_set"], 1)
            self.assertEqual(saved["status"], "done")
            self.assertIsNotNone(saved["best_params"])
            self.assertIsNotNone(saved["best_value"])
            self.assertEqual(saved["n_trials"], 3)
            self.assertEqual(len(saved["trials"]), 3)

    def test_skips_existing_study(self):
        """If best_params.json exists, study is skipped (no optuna needed)."""
        from lg_cotrain.optuna_per_experiment import run_single_study

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create best_params.json
            out_dir = Path(tmpdir) / "test_event" / "5_set1"
            out_dir.mkdir(parents=True)
            existing = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 10, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(existing))

            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=15, storage_dir=tmpdir,
            )

            # Should return existing data
            self.assertEqual(result["best_value"], 0.85)


class TestStudyWorkerPicklable(unittest.TestCase):
    """Verify _run_study_worker receives a plain dict."""

    def test_worker_receives_dict(self):
        """Worker function signature accepts a dict."""
        from lg_cotrain.optuna_per_experiment import _run_study_worker
        import inspect

        sig = inspect.signature(_run_study_worker)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["kwargs"])

        # Check the annotation is dict
        param = sig.parameters["kwargs"]
        self.assertEqual(param.annotation, dict)


class TestGPUAssignmentForStudies(unittest.TestCase):
    """Verify round-robin GPU assignment across studies."""

    def test_2_gpus_6_studies(self):
        configs = [{"event": "e", "budget": b, "seed_set": s}
                   for b in [5, 10] for s in [1, 2, 3]]
        num_gpus = 2
        for i, cfg in enumerate(configs):
            cfg["device"] = f"cuda:{i % num_gpus}"
        devices = [c["device"] for c in configs]
        self.assertEqual(devices, [
            "cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0", "cuda:1",
        ])

    def test_3_gpus_3_studies(self):
        configs = [{"event": "e", "budget": 5, "seed_set": s}
                   for s in [1, 2, 3]]
        num_gpus = 3
        for i, cfg in enumerate(configs):
            cfg["device"] = f"cuda:{i % num_gpus}"
        devices = [c["device"] for c in configs]
        self.assertEqual(devices, ["cuda:0", "cuda:1", "cuda:2"])


class TestAllStudiesSkipsCompleted(unittest.TestCase):
    """Verify run_all_studies skips experiments with existing best_params.json."""

    def test_all_skipped_no_worker_call(self):
        from lg_cotrain.optuna_per_experiment import run_all_studies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create all results for a small grid
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}"
                out_dir.mkdir(parents=True)
                result = {
                    "event": "test_event", "budget": 5, "seed_set": seed_set,
                    "status": "done", "best_params": {"lr": 0.001},
                    "best_value": 0.85, "n_trials": 10, "trials": [],
                }
                (out_dir / "best_params.json").write_text(json.dumps(result))

            with patch("lg_cotrain.optuna_per_experiment._run_study_worker") as mock_worker:
                results = run_all_studies(
                    events=["test_event"],
                    budgets=[5],
                    seed_sets=[1, 2],
                    n_trials=10,
                    storage_dir=tmpdir,
                )

            mock_worker.assert_not_called()
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r["status"] == "done" for r in results))

    def test_partial_skip(self):
        """Only pending studies should be run."""
        from lg_cotrain.optuna_per_experiment import run_all_studies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create seed_set=1 only
            out_dir = Path(tmpdir) / "test_event" / "5_set1"
            out_dir.mkdir(parents=True)
            existing = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 10, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(existing))

            new_result = {
                "event": "test_event", "budget": 5, "seed_set": 2,
                "status": "done", "best_params": {"lr": 0.0005},
                "best_value": 0.82, "n_trials": 10, "trials": [],
            }

            with patch("lg_cotrain.optuna_per_experiment._run_study_worker") as mock_worker:
                mock_worker.return_value = new_result
                results = run_all_studies(
                    events=["test_event"],
                    budgets=[5],
                    seed_sets=[1, 2],
                    n_trials=10,
                    storage_dir=tmpdir,
                )

            # Only 1 study should have been dispatched
            self.assertEqual(mock_worker.call_count, 1)
            # Both results should be present
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["best_value"], 0.85)  # skipped
            self.assertEqual(results[1]["best_value"], 0.82)  # ran

    def test_summary_json_written(self):
        """summary.json should be written after all studies complete."""
        from lg_cotrain.optuna_per_experiment import run_all_studies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create all results
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}"
                out_dir.mkdir(parents=True)
                result = {
                    "event": "test_event", "budget": 5, "seed_set": seed_set,
                    "status": "done", "best_params": {"lr": 0.001},
                    "best_value": 0.85, "n_trials": 10, "trials": [],
                }
                (out_dir / "best_params.json").write_text(json.dumps(result))

            run_all_studies(
                events=["test_event"],
                budgets=[5],
                seed_sets=[1, 2],
                n_trials=10,
                storage_dir=tmpdir,
            )

            summary_path = Path(tmpdir) / "summary.json"
            self.assertTrue(summary_path.exists())

            with open(summary_path) as f:
                summary = json.load(f)

            self.assertEqual(summary["total_studies"], 2)
            self.assertEqual(len(summary["studies"]), 2)


class TestLoadBestParams(unittest.TestCase):
    """Verify load_best_params reads correct files."""

    def test_loads_existing(self):
        from lg_cotrain.optuna_per_experiment import load_best_params

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 best_params.json files
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}"
                out_dir.mkdir(parents=True)
                data = {
                    "event": "test_event", "budget": 5, "seed_set": seed_set,
                    "status": "done", "best_params": {"lr": 0.001 * seed_set},
                    "best_value": 0.80 + seed_set * 0.01,
                    "n_trials": 10, "trials": [],
                }
                (out_dir / "best_params.json").write_text(json.dumps(data))

            results = load_best_params(
                storage_dir=tmpdir,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1, 2],
            )

            self.assertEqual(len(results), 2)
            self.assertIn(("test_event", 5, 1), results)
            self.assertIn(("test_event", 5, 2), results)
            self.assertEqual(results[("test_event", 5, 1)]["best_params"]["lr"], 0.001)
            self.assertEqual(results[("test_event", 5, 2)]["best_params"]["lr"], 0.002)

    def test_missing_files_skipped(self):
        """Missing best_params.json files are silently skipped."""
        from lg_cotrain.optuna_per_experiment import load_best_params

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create seed_set=1
            out_dir = Path(tmpdir) / "test_event" / "5_set1"
            out_dir.mkdir(parents=True)
            data = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 10, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(data))

            results = load_best_params(
                storage_dir=tmpdir,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1, 2, 3],
            )

            self.assertEqual(len(results), 1)
            self.assertIn(("test_event", 5, 1), results)
            self.assertNotIn(("test_event", 5, 2), results)


class TestCLIFlags(unittest.TestCase):
    """CLI flags are accepted and forwarded."""

    def test_help_flag(self):
        """--help should not crash."""
        from lg_cotrain.optuna_per_experiment import main
        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.argv", ["prog", "--help"]):
                main()
        self.assertEqual(ctx.exception.code, 0)

    def test_n_trials_forwarded(self):
        from lg_cotrain.optuna_per_experiment import main

        with patch("lg_cotrain.optuna_per_experiment.run_all_studies") as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", [
                "prog", "--n-trials", "25", "--num-gpus", "2",
                "--events", "test_event",
            ]):
                main()

            call_kwargs = mock_run.call_args
            self.assertEqual(call_kwargs.kwargs.get("n_trials"), 25)
            self.assertEqual(call_kwargs.kwargs.get("num_gpus"), 2)
            self.assertEqual(call_kwargs.kwargs.get("events"), ["test_event"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
