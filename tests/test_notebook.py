"""Tests for the experiment notebook structure and logic.

Validates notebook JSON structure, cell ordering, and that the notebook's
helper functions and data preparation steps match the trainer.py implementation.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, "/workspace")

NOTEBOOK_PATH = "/workspace/Notebooks/01_kaikoura_experiment.ipynb"


class TestNotebookStructure(unittest.TestCase):
    """Test the notebook file structure and cell layout."""

    def setUp(self):
        with open(NOTEBOOK_PATH) as f:
            self.nb = json.load(f)

    def test_notebook_is_valid_ipynb(self):
        """Notebook has required top-level keys for a valid .ipynb file."""
        self.assertIn("cells", self.nb)
        self.assertIn("metadata", self.nb)
        self.assertIn("nbformat", self.nb)
        self.assertEqual(self.nb["nbformat"], 4)

    def test_cell_count(self):
        """Notebook has the expected number of cells (34)."""
        self.assertEqual(len(self.nb["cells"]), 34)

    def test_cell_types_alternate_correctly(self):
        """Verify the expected pattern of markdown and code cells."""
        expected_types = [
            "markdown",  # 1: Title
            "markdown",  # 2: Env setup explanation
            "code",      # 3: Imports
            "markdown",  # 4: Config explanation
            "code",      # 5: Config + seed + device
            "markdown",  # 6: Data loading explanation
            "code",      # 7: Load data
            "code",      # 8: Class distributions
            "markdown",  # 9: Data prep explanation
            "code",      # 10: Label encoder, split, D_LG
            "markdown",  # 11: Tokenization explanation
            "code",      # 12: Datasets and dataloaders
            "markdown",  # 13: collect_probs explanation
            "code",      # 14: collect_probs function
            "markdown",  # 15: Phase 1 explanation
            "code",      # 16: Phase 1 loop
            "markdown",  # 17: Phase 1 results analysis
            "code",      # 18: Phase 1 visualizations
            "markdown",  # 19: Phase 2 explanation
            "code",      # 20: Phase 2 loop
            "code",      # 21: Phase 2 visualizations
            "markdown",  # 22: Phase 3 explanation
            "code",      # 23: Phase 3 loop
            "code",      # 24: Phase 3 visualizations
            "markdown",  # 25: Final eval explanation
            "code",      # 26: Final evaluation
            "code",      # 27: Per-class F1 bar chart
            "code",      # 28: Save results
            "markdown",  # 29: Section 11 header
            "code",      # 30: run_all_experiments
            "code",      # 31: print summary table
            "markdown",  # 32: Results by Budget header
            "code",      # 33: Budget visualization
            "markdown",  # 34: Summary
        ]
        actual_types = [cell["cell_type"] for cell in self.nb["cells"]]
        self.assertEqual(actual_types, expected_types)

    def test_all_cells_have_source(self):
        """Every cell has a non-empty source field."""
        for i, cell in enumerate(self.nb["cells"]):
            source = "".join(cell["source"])
            self.assertTrue(
                len(source.strip()) > 0,
                f"Cell {i+1} has empty source",
            )

    def test_markdown_cells_have_headers(self):
        """Each markdown cell starts with a heading or meaningful content."""
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "markdown":
                source = "".join(cell["source"]).strip()
                self.assertTrue(
                    len(source) > 10,
                    f"Markdown cell {i+1} is too short: {source[:50]}",
                )

    def test_code_cells_have_no_outputs(self):
        """Code cells have no pre-filled outputs (clean notebook)."""
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "code":
                outputs = cell.get("outputs", [])
                self.assertEqual(
                    len(outputs), 0,
                    f"Code cell {i+1} has pre-filled outputs",
                )


class TestNotebookImports(unittest.TestCase):
    """Test that the notebook imports all required modules."""

    def setUp(self):
        with open(NOTEBOOK_PATH) as f:
            self.nb = json.load(f)
        # Find the imports cell (cell index 2, which is the 3rd cell)
        self.imports_source = "".join(self.nb["cells"][2]["source"])

    def test_imports_sys_path(self):
        """Notebook dynamically adds repo root to sys.path."""
        self.assertIn("_find_repo_root", self.imports_source)
        self.assertIn("sys.path.insert(0,", self.imports_source)

    def test_imports_torch(self):
        self.assertIn("import torch", self.imports_source)

    def test_imports_config(self):
        self.assertIn("from lg_cotrain.config import LGCoTrainConfig", self.imports_source)

    def test_imports_data_loading(self):
        self.assertIn("from lg_cotrain.data_loading import", self.imports_source)
        self.assertIn("CLASS_LABELS", self.imports_source)
        self.assertIn("TweetDataset", self.imports_source)
        self.assertIn("build_d_lg", self.imports_source)
        self.assertIn("build_label_encoder", self.imports_source)
        self.assertIn("detect_event_classes", self.imports_source)
        self.assertIn("load_pseudo_labels", self.imports_source)
        self.assertIn("load_tsv", self.imports_source)
        self.assertIn("split_labeled_set", self.imports_source)

    def test_imports_evaluate(self):
        self.assertIn("from lg_cotrain.evaluate import", self.imports_source)
        self.assertIn("compute_ece", self.imports_source)
        self.assertIn("compute_metrics", self.imports_source)
        self.assertIn("ensemble_predict", self.imports_source)

    def test_imports_model(self):
        self.assertIn("from lg_cotrain.model import create_fresh_model", self.imports_source)

    def test_imports_utils(self):
        self.assertIn("from lg_cotrain.utils import", self.imports_source)
        self.assertIn("EarlyStopping", self.imports_source)
        self.assertIn("get_device", self.imports_source)
        self.assertIn("set_seed", self.imports_source)

    def test_imports_weight_tracker(self):
        self.assertIn("from lg_cotrain.weight_tracker import WeightTracker", self.imports_source)

    def test_imports_matplotlib(self):
        self.assertIn("import matplotlib.pyplot as plt", self.imports_source)


class TestNotebookContent(unittest.TestCase):
    """Test that key pipeline steps are present in the notebook code cells."""

    def setUp(self):
        with open(NOTEBOOK_PATH) as f:
            self.nb = json.load(f)
        self.code_sources = {}
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "code":
                self.code_sources[i] = "".join(cell["source"])
        self.all_code = "\n".join(self.code_sources.values())

    def test_config_uses_kaikoura(self):
        """Config cell uses kaikoura_earthquake_2016 event."""
        config_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("kaikoura_earthquake_2016", config_cell)
        self.assertIn("budget=5", config_cell)
        self.assertIn("seed_set=1", config_cell)

    def test_collect_probs_defined(self):
        """The collect_probs helper function is defined."""
        self.assertIn("def collect_probs(model, loader, pseudo_label_ids)", self.all_code)

    def test_collect_probs_uses_predict_proba(self):
        """collect_probs calls model.predict_proba."""
        collect_cell = "".join(self.nb["cells"][13]["source"])
        self.assertIn("predict_proba", collect_cell)

    def test_phase1_creates_fresh_models(self):
        """Phase 1 creates fresh models and weight trackers."""
        phase1_cell = "".join(self.nb["cells"][15]["source"])
        self.assertIn("create_fresh_model", phase1_cell)
        self.assertIn("WeightTracker", phase1_cell)
        self.assertIn("tracker1.record_epoch", phase1_cell)
        self.assertIn("tracker2.record_epoch", phase1_cell)
        self.assertIn("compute_lambda_optimistic", phase1_cell)
        self.assertIn("compute_lambda_conservative", phase1_cell)

    def test_phase2_uses_weighted_ce(self):
        """Phase 2 uses weighted cross-entropy loss."""
        phase2_cell = "".join(self.nb["cells"][19]["source"])
        self.assertIn("cross_entropy", phase2_cell)
        self.assertIn("reduction=\"none\"", phase2_cell)
        self.assertIn("lambda2[sample_idx]", phase2_cell)
        self.assertIn("lambda1[sample_idx]", phase2_cell)

    def test_phase2_seeds_from_phase1(self):
        """Phase 2 seeds its trackers from Phase 1's full history."""
        phase2_cell = "".join(self.nb["cells"][19]["source"])
        self.assertIn("seed_from_tracker(tracker1)", phase2_cell)
        self.assertIn("seed_from_tracker(tracker2)", phase2_cell)

    def test_phase3_uses_early_stopping(self):
        """Phase 3 uses EarlyStopping and restores best models."""
        phase3_cell = "".join(self.nb["cells"][22]["source"])
        self.assertIn("EarlyStopping", phase3_cell)
        self.assertIn("es1.step(", phase3_cell)
        self.assertIn("es2.step(", phase3_cell)
        self.assertIn("es1.restore_best", phase3_cell)
        self.assertIn("es2.restore_best", phase3_cell)

    def test_final_eval_uses_ensemble(self):
        """Final evaluation uses ensemble_predict and compute_ece."""
        eval_cell = "".join(self.nb["cells"][25]["source"])
        self.assertIn("ensemble_predict", eval_cell)
        self.assertIn("compute_metrics", eval_cell)
        self.assertIn("compute_ece", eval_cell)

    def test_results_saved_to_json(self):
        """Results are saved to metrics.json."""
        save_cell = "".join(self.nb["cells"][27]["source"])
        self.assertIn("metrics.json", save_cell)
        self.assertIn("json.dump", save_cell)
        self.assertIn("test_error_rate", save_cell)
        self.assertIn("test_macro_f1", save_cell)
        self.assertIn("test_ece", save_cell)
        self.assertIn("test_per_class_f1", save_cell)
        self.assertIn("dev_error_rate", save_cell)
        self.assertIn("dev_macro_f1", save_cell)
        self.assertIn("dev_ece", save_cell)
        self.assertIn("lambda1_mean", save_cell)
        self.assertIn("lambda2_mean", save_cell)

    def test_visualizations_present(self):
        """Visualization cells use matplotlib."""
        # Phase 1 viz (cell 17)
        viz1 = "".join(self.nb["cells"][17]["source"])
        self.assertIn("plt.show()", viz1)

        # Phase 2 viz (cell 20)
        viz2 = "".join(self.nb["cells"][20]["source"])
        self.assertIn("plt.show()", viz2)

        # Phase 3 viz (cell 23)
        viz3 = "".join(self.nb["cells"][23]["source"])
        self.assertIn("plt.show()", viz3)

        # Per-class F1 viz (cell 26)
        viz4 = "".join(self.nb["cells"][26]["source"])
        self.assertIn("plt.show()", viz4)

    def test_tracking_lists_for_plots(self):
        """Phase tracking lists are accumulated for visualization."""
        self.assertIn("phase1_mean_probs1", self.all_code)
        self.assertIn("phase1_mean_probs2", self.all_code)
        self.assertIn("phase2_losses1", self.all_code)
        self.assertIn("phase2_losses2", self.all_code)
        self.assertIn("phase2_dev_f1s", self.all_code)
        self.assertIn("phase3_dev_f1s", self.all_code)


class TestNotebookMatchesTrainer(unittest.TestCase):
    """Test that the notebook logic matches trainer.py's implementation."""

    def setUp(self):
        with open(NOTEBOOK_PATH) as f:
            self.nb = json.load(f)
        self.all_code = "\n".join(
            "".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code"
        )

    def test_phase2_model1_uses_lambda2_weights(self):
        """Model 1's loss in Phase 2 is weighted by lambda2 (from model 2)."""
        phase2_code = "".join(self.nb["cells"][19]["source"])
        # Model 1 uses w2 (lambda2) weights
        self.assertIn("w2 * per_sample_loss1", phase2_code)
        # Model 2 uses w1 (lambda1) weights
        self.assertIn("w1 * per_sample_loss2", phase2_code)

    def test_results_format_matches_trainer(self):
        """Results dict has the same keys as trainer.py output."""
        save_cell = "".join(self.nb["cells"][27]["source"])
        required_keys = [
            "event", "budget", "seed_set",
            "test_error_rate", "test_macro_f1", "test_ece", "test_per_class_f1",
            "dev_error_rate", "dev_macro_f1", "dev_ece",
            "lambda1_mean", "lambda1_std",
            "lambda2_mean", "lambda2_std",
        ]
        for key in required_keys:
            self.assertIn(f'"{key}"', save_cell, f"Missing key: {key}")

    def test_phase2_creates_fresh_models(self):
        """Phase 2 creates fresh models (not reusing Phase 1 models)."""
        phase2_code = "".join(self.nb["cells"][19]["source"])
        self.assertIn("create_fresh_model(cfg)", phase2_code)

    def test_data_loading_matches_trainer(self):
        """Data loading uses the same functions as trainer.py."""
        load_cell = "".join(self.nb["cells"][6]["source"])
        self.assertIn("load_tsv(cfg.labeled_path)", load_cell)
        self.assertIn("load_tsv(cfg.unlabeled_path)", load_cell)
        self.assertIn("load_pseudo_labels(cfg.pseudo_label_path)", load_cell)
        self.assertIn("load_tsv(cfg.dev_path)", load_cell)
        self.assertIn("load_tsv(cfg.test_path)", load_cell)

    def test_data_prep_matches_trainer(self):
        """Data preparation uses the same functions and order as trainer.py."""
        prep_cell = "".join(self.nb["cells"][9]["source"])
        self.assertIn("detect_event_classes(", prep_cell)
        self.assertIn("build_label_encoder(", prep_cell)
        self.assertIn("split_labeled_set(df_labeled", prep_cell)
        self.assertIn("build_d_lg(df_unlabeled, df_pseudo)", prep_cell)


class TestNotebookDataFiles(unittest.TestCase):
    """Test that the data files referenced by the notebook exist."""

    def setUp(self):
        from lg_cotrain.config import LGCoTrainConfig
        self.cfg = LGCoTrainConfig(
            event="kaikoura_earthquake_2016",
            budget=5,
            seed_set=1,
        )

    def test_labeled_file_exists(self):
        self.assertTrue(os.path.exists(self.cfg.labeled_path), self.cfg.labeled_path)

    def test_unlabeled_file_exists(self):
        self.assertTrue(os.path.exists(self.cfg.unlabeled_path), self.cfg.unlabeled_path)

    def test_pseudo_label_file_exists(self):
        self.assertTrue(os.path.exists(self.cfg.pseudo_label_path), self.cfg.pseudo_label_path)

    def test_dev_file_exists(self):
        self.assertTrue(os.path.exists(self.cfg.dev_path), self.cfg.dev_path)

    def test_test_file_exists(self):
        self.assertTrue(os.path.exists(self.cfg.test_path), self.cfg.test_path)


NOTEBOOK_03_PATH = "/workspace/Notebooks/03_all_disasters_rerun.ipynb"


class TestNotebook03Structure(unittest.TestCase):
    """Test the notebook 03 file structure and cell layout."""

    def setUp(self):
        with open(NOTEBOOK_03_PATH) as f:
            self.nb = json.load(f)

    def test_notebook_is_valid_ipynb(self):
        """Notebook has required top-level keys for a valid .ipynb file."""
        self.assertIn("cells", self.nb)
        self.assertIn("metadata", self.nb)
        self.assertIn("nbformat", self.nb)
        self.assertEqual(self.nb["nbformat"], 4)

    def test_cell_count(self):
        """Notebook has the expected number of cells (10)."""
        self.assertEqual(len(self.nb["cells"]), 10)

    def test_cell_types(self):
        """Verify the expected pattern of markdown and code cells."""
        expected_types = [
            "markdown",  # 0: Title + explanation
            "code",      # 1: Setup — imports, repo root
            "code",      # 2: Configuration cell
            "code",      # 3: Event discovery + completion check
            "markdown",  # 4: Running experiments explanation
            "code",      # 5: Main experiment loop
            "markdown",  # 6: Cross-disaster results explanation
            "code",      # 7: Cross-disaster summary + plots
            "code",      # 8: Generate dashboard
            "markdown",  # 9: Summary with CLI equivalents
        ]
        actual_types = [cell["cell_type"] for cell in self.nb["cells"]]
        self.assertEqual(actual_types, expected_types)

    def test_all_cells_have_source(self):
        """Every cell has a non-empty source field."""
        for i, cell in enumerate(self.nb["cells"]):
            source = "".join(cell["source"])
            self.assertTrue(
                len(source.strip()) > 0,
                f"Cell {i} has empty source",
            )

    def test_code_cells_have_no_outputs(self):
        """Code cells have no pre-filled outputs (clean notebook)."""
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "code":
                outputs = cell.get("outputs", [])
                self.assertEqual(
                    len(outputs), 0,
                    f"Code cell {i} has pre-filled outputs",
                )


class TestNotebook03Config(unittest.TestCase):
    """Test that notebook 03 uses configurable pseudo-label source and run name."""

    def setUp(self):
        with open(NOTEBOOK_03_PATH) as f:
            self.nb = json.load(f)
        self.all_code = "\n".join(
            "".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code"
        )

    def test_config_has_pseudo_label_source(self):
        """Configuration cell defines PSEUDO_LABEL_SOURCE."""
        config_cell = "".join(self.nb["cells"][2]["source"])
        self.assertIn("PSEUDO_LABEL_SOURCE", config_cell)

    def test_config_has_run_name(self):
        """Configuration cell defines RUN_NAME."""
        config_cell = "".join(self.nb["cells"][2]["source"])
        self.assertIn("RUN_NAME", config_cell)

    def test_results_root_uses_run_name(self):
        """RESULTS_ROOT includes the RUN_NAME sub-folder."""
        config_cell = "".join(self.nb["cells"][2]["source"])
        self.assertIn("results\" / RUN_NAME", config_cell)

    def test_pseudo_label_source_passed_to_run_all(self):
        """Experiment loop passes pseudo_label_source=PSEUDO_LABEL_SOURCE."""
        experiment_cell = "".join(self.nb["cells"][5]["source"])
        self.assertIn("pseudo_label_source=PSEUDO_LABEL_SOURCE", experiment_cell)

    def test_results_root_passed_to_run_all(self):
        """Experiment loop passes results_root=RESULTS_ROOT."""
        experiment_cell = "".join(self.nb["cells"][5]["source"])
        self.assertIn("results_root=RESULTS_ROOT", experiment_cell)


class TestNotebook03Imports(unittest.TestCase):
    """Test that notebook 03 imports required modules."""

    def setUp(self):
        with open(NOTEBOOK_03_PATH) as f:
            self.nb = json.load(f)
        self.imports_source = "".join(self.nb["cells"][1]["source"])

    def test_imports_sys_path(self):
        """Notebook dynamically adds repo root to sys.path."""
        self.assertIn("_find_repo_root", self.imports_source)
        self.assertIn("sys.path.insert(0,", self.imports_source)

    def test_imports_run_all(self):
        """Notebook imports run_all_experiments and format_summary_table."""
        self.assertIn("run_all_experiments", self.imports_source)
        self.assertIn("format_summary_table", self.imports_source)

    def test_imports_budgets_and_seeds(self):
        """Notebook imports BUDGETS and SEED_SETS constants."""
        self.assertIn("BUDGETS", self.imports_source)
        self.assertIn("SEED_SETS", self.imports_source)

    def test_imports_matplotlib(self):
        self.assertIn("import matplotlib.pyplot as plt", self.imports_source)


class TestNotebook03Content(unittest.TestCase):
    """Test that notebook 03 has the expected experiment and dashboard logic."""

    def setUp(self):
        with open(NOTEBOOK_03_PATH) as f:
            self.nb = json.load(f)
        self.all_code = "\n".join(
            "".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code"
        )

    def test_event_discovery(self):
        """Notebook discovers events from data directory."""
        discovery_cell = "".join(self.nb["cells"][3]["source"])
        self.assertIn("is_event_complete", discovery_cell)
        self.assertIn("completed_events", discovery_cell)
        self.assertIn("pending_events", discovery_cell)

    def test_progress_tracker(self):
        """Notebook uses ProgressTracker for monitoring."""
        exp_cell = "".join(self.nb["cells"][5]["source"])
        self.assertIn("ProgressTracker", exp_cell)
        self.assertIn("_on_experiment_done=tracker.update", exp_cell)

    def test_cross_disaster_summary(self):
        """Notebook builds cross-disaster summary."""
        summary_cell = "".join(self.nb["cells"][7]["source"])
        self.assertIn("summary", summary_cell)
        self.assertIn("test_macro_f1", summary_cell)

    def test_visualizations_present(self):
        """Notebook has plots."""
        summary_cell = "".join(self.nb["cells"][7]["source"])
        self.assertIn("plt.show()", summary_cell)

    def test_dashboard_generation(self):
        """Notebook generates an HTML dashboard."""
        dashboard_cell = "".join(self.nb["cells"][8]["source"])
        self.assertIn("collect_all_metrics", dashboard_cell)
        self.assertIn("generate_html", dashboard_cell)
        self.assertIn("dashboard.html", dashboard_cell)
        self.assertIn("RESULTS_ROOT", dashboard_cell)

    def test_summary_shows_cli_equivalent(self):
        """Summary markdown includes CLI equivalent commands."""
        summary_md = "".join(self.nb["cells"][9]["source"])
        self.assertIn("--pseudo-label-source", summary_md)
        self.assertIn("--output-folder", summary_md)
        self.assertIn("run_experiment", summary_md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
