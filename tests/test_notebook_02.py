"""Tests for the all-disasters experiment notebook (02)."""

import json
import sys
import unittest

sys.path.insert(0, "/workspace")

NOTEBOOK_PATH = "/workspace/Notebooks/02_all_disasters_experiment.ipynb"


class TestNotebook02Structure(unittest.TestCase):
    """Test the notebook file structure and cell layout."""

    def setUp(self):
        with open(NOTEBOOK_PATH) as f:
            self.nb = json.load(f)

    def test_notebook_is_valid_ipynb(self):
        self.assertIn("cells", self.nb)
        self.assertIn("metadata", self.nb)
        self.assertIn("nbformat", self.nb)
        self.assertEqual(self.nb["nbformat"], 4)

    def test_cell_count(self):
        """Notebook has the expected number of cells (10)."""
        self.assertEqual(len(self.nb["cells"]), 10)

    def test_cell_types(self):
        expected_types = [
            "markdown",  # 0: Title & overview
            "code",      # 1: Environment setup
            "code",      # 2: Discover events & check completion
            "markdown",  # 3: Running experiments explanation
            "code",      # 4: Run all pending events
            "markdown",  # 5: Cross-disaster results explanation
            "code",      # 6: Cross-disaster summary table
            "code",      # 7: Line plot visualization
            "code",      # 8: Heatmap visualization
            "markdown",  # 9: Summary
        ]
        actual_types = [cell["cell_type"] for cell in self.nb["cells"]]
        self.assertEqual(actual_types, expected_types)

    def test_all_cells_have_source(self):
        for i, cell in enumerate(self.nb["cells"]):
            source = "".join(cell["source"])
            self.assertTrue(
                len(source.strip()) > 0,
                f"Cell {i} has empty source",
            )

    def test_code_cells_have_no_outputs(self):
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "code":
                outputs = cell.get("outputs", [])
                self.assertEqual(
                    len(outputs), 0,
                    f"Code cell {i} has pre-filled outputs",
                )

    def test_markdown_cells_have_headers(self):
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "markdown":
                source = "".join(cell["source"]).strip()
                self.assertTrue(
                    len(source) > 10,
                    f"Markdown cell {i} is too short",
                )


class TestNotebook02Content(unittest.TestCase):
    """Test that key logic is present in notebook code cells."""

    def setUp(self):
        with open(NOTEBOOK_PATH) as f:
            self.nb = json.load(f)
        self.all_code = "\n".join(
            "".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code"
        )

    def test_imports_run_all(self):
        setup_cell = "".join(self.nb["cells"][1]["source"])
        self.assertIn("from lg_cotrain.run_all import", setup_cell)
        self.assertIn("run_all_experiments", setup_cell)
        self.assertIn("format_summary_table", setup_cell)
        self.assertIn("BUDGETS", setup_cell)
        self.assertIn("SEED_SETS", setup_cell)

    def test_imports_matplotlib(self):
        setup_cell = "".join(self.nb["cells"][1]["source"])
        self.assertIn("import matplotlib.pyplot as plt", setup_cell)

    def test_repo_root_discovery(self):
        setup_cell = "".join(self.nb["cells"][1]["source"])
        self.assertIn("_find_repo_root", setup_cell)
        self.assertIn("sys.path.insert(0,", setup_cell)

    def test_event_discovery(self):
        discover_cell = "".join(self.nb["cells"][2]["source"])
        self.assertIn("data", discover_cell)
        self.assertIn("original", discover_cell)
        self.assertIn("is_dir()", discover_cell)

    def test_is_event_complete_defined(self):
        discover_cell = "".join(self.nb["cells"][2]["source"])
        self.assertIn("def is_event_complete(", discover_cell)
        self.assertIn("metrics.json", discover_cell)

    def test_completed_pending_partition(self):
        discover_cell = "".join(self.nb["cells"][2]["source"])
        self.assertIn("completed_events", discover_cell)
        self.assertIn("pending_events", discover_cell)

    def test_run_loop_calls_run_all(self):
        run_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("run_all_experiments(", run_cell)
        self.assertIn("format_summary_table(", run_cell)

    def test_run_loop_iterates_pending(self):
        run_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("pending_events", run_cell)

    def test_completed_events_loaded(self):
        """Results for already-completed events are loaded from metrics.json."""
        run_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("completed_events", run_cell)
        self.assertIn("json.load", run_cell)

    def test_progress_tracker_defined(self):
        """Cell 4 defines a ProgressTracker class for time tracking."""
        run_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("class ProgressTracker", run_cell)
        self.assertIn("def update(", run_cell)

    def test_progress_callback_passed(self):
        """Cell 4 passes a callback to run_all_experiments."""
        run_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("_on_experiment_done", run_cell)

    def test_progress_displays_eta(self):
        """Progress tracker prints percentage and ETA."""
        run_cell = "".join(self.nb["cells"][4]["source"])
        self.assertIn("PROGRESS", run_cell)
        self.assertIn("ETA", run_cell)

    def test_cross_disaster_summary(self):
        summary_cell = "".join(self.nb["cells"][6]["source"])
        self.assertIn("all_event_results", summary_cell)
        self.assertIn("f1_mean", summary_cell)

    def test_line_plot_visualization(self):
        viz_cell = "".join(self.nb["cells"][7]["source"])
        self.assertIn("plt.show()", viz_cell)
        self.assertIn("errorbar", viz_cell)

    def test_heatmap_visualization(self):
        viz_cell = "".join(self.nb["cells"][8]["source"])
        self.assertIn("plt.show()", viz_cell)
        self.assertIn("imshow", viz_cell)

    def test_all_10_events_listed_in_title(self):
        """Title markdown lists all 10 events."""
        title_cell = "".join(self.nb["cells"][0]["source"])
        expected_events = [
            "california_wildfires_2018",
            "canada_wildfires_2016",
            "cyclone_idai_2019",
            "hurricane_dorian_2019",
            "hurricane_florence_2018",
            "hurricane_harvey_2017",
            "hurricane_irma_2017",
            "hurricane_maria_2017",
            "kaikoura_earthquake_2016",
            "kerala_floods_2018",
        ]
        for event in expected_events:
            self.assertIn(event, title_cell)


if __name__ == "__main__":
    unittest.main(verbosity=2)
