# LG-CoTrain

**LLM-Guided Co-Training for Humanitarian Tweet Classification**

> **Results Dashboard** — View experiment results: [results/dashboard.html](https://htmlpreview.github.io/?https://github.com/anhtranst/Co-training/blob/main/results/dashboard.html)
> _(Open locally in a browser; rebuild anytime with `python -m lg_cotrain.dashboard`)_

A semi-supervised co-training pipeline that classifies humanitarian tweets during disaster events. It combines a small set of human-labeled tweets with LLM pseudo-labeled tweets (e.g., from GPT-4o) using a 3-phase training approach with two BERT models.

---

## Table of Contents

- [Motivation](#motivation)
- [How It Works](#how-it-works)
  - [Phase 1 — Weight Generation](#phase-1--weight-generation)
  - [Phase 2 — Co-Training](#phase-2--co-training)
  - [Phase 3 — Fine-Tuning](#phase-3--fine-tuning)
- [Class Labels](#class-labels)
- [Disaster Events](#disaster-events)
- [Data Layout](#data-layout)
- [Installation](#installation)
- [Usage](#usage)
  - [Single Experiment](#single-experiment)
  - [Batch Mode (Multiple Events/Budgets/Seeds)](#batch-mode-multiple-eventsbudgetsseeds)
  - [Custom Pseudo-Label Source and Output Folder](#custom-pseudo-label-source-and-output-folder)
  - [Interactive Notebooks](#interactive-notebooks)
  - [Results Dashboard](#results-dashboard)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [References](#references)

---

## Motivation

During disasters, rapid classification of social media posts helps humanitarian organizations prioritize response efforts. However, manually labeling enough tweets to train a reliable classifier is slow and expensive.

LG-CoTrain addresses this by:

1. Starting with a **small set of human-labeled tweets** (as few as 5 per class)
2. Using an **LLM (e.g., GPT-4o) to pseudo-label** a large pool of unlabeled tweets
3. Computing **per-sample reliability weights** that measure how trustworthy each pseudo-label is
4. **Co-training two BERT classifiers** that teach each other using these weighted pseudo-labels
5. **Fine-tuning** on the small labeled set with early stopping

The result is a classifier that significantly outperforms training on labeled data alone, even when labeled data is extremely scarce.

---

## How It Works

The pipeline has three phases, each building on the previous one. Two BERT models work together throughout, exchanging information about which pseudo-labels they trust.

```
                        ┌─────────────────────────────────────────────────┐
                        │              INPUT DATA                         │
                        │                                                 │
                        │  D_labeled ──► split ──► D_l1 (half 1)         │
                        │                     └──► D_l2 (half 2)         │
                        │                                                 │
                        │  D_unlabeled + LLM pseudo-labels ──► D_LG      │
                        │  D_dev  (development set for early stopping)    │
                        │  D_test (held-out test set)                     │
                        └─────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┬┘
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │   BERT Model 1    │   │   BERT Model 2    │
        │   (trained on     │   │   (trained on     │
        │    D_l1)          │   │    D_l2)          │
        └────────┬──────────┘   └──────────┬────────┘
                 │                          │
                 │  ┌──────────────────┐    │
                 └─►│  PHASE 1         │◄───┘
                    │  Weight          │
                    │  Generation      │
                    │  (7 epochs)      │
                    └────────┬─────────┘
                             │
                  Compute lambda weights
                  from probability history
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  lambda1          │          │  lambda2          │
    │  (optimistic)     │          │  (conservative)   │
    │  = conf + var     │          │  = conf - var     │
    └────────┬─────────┘          └─────────┬────────┘
             │                               │
             │  ┌──────────────────────┐     │
             │  │  PHASE 2             │     │
             └─►│  Co-Training         │◄────┘
                │  (10 epochs)         │
                │                      │
                │  Model 1 trained     │
                │  with lambda2 wts    │
                │  (from Model 2)      │
                │                      │
                │  Model 2 trained     │
                │  with lambda1 wts    │
                │  (from Model 1)      │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                          ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  PHASE 3          │      │  PHASE 3          │
    │  Fine-Tune        │      │  Fine-Tune        │
    │  Model 1 on D_l1  │      │  Model 2 on D_l2  │
    │  (early stopping  │      │  (early stopping   │
    │   on dev F1)      │      │   on dev F1)       │
    └────────┬─────────┘      └─────────┬─────────┘
             │                           │
             └─────────┬─────────────────┘
                       ▼
              ┌──────────────────┐
              │  ENSEMBLE         │
              │  Average softmax  │
              │  from both models │
              │  then argmax      │
              └──────────────────┘
                       │
                       ▼
                 Final Predictions
```

### Phase 1 — Weight Generation

**Goal**: Learn how much each pseudo-label can be trusted.

Two fresh BERT models are trained **separately** — Model 1 on D_l1, Model 2 on D_l2 (stratified halves of the small labeled set). After each training epoch, both models predict softmax probabilities for every sample in the pseudo-labeled set (D_LG). These predictions are recorded by a `WeightTracker`.

After all epochs, lambda weights are computed per sample:

```
confidence  = mean of p(pseudo_label | x; theta) across all epochs
variability = std  of p(pseudo_label | x; theta) across all epochs

lambda_optimistic (lambda1)    = confidence + variability
lambda_conservative (lambda2)  = max(confidence - variability, 0)
```

**Intuition**:

- **High confidence, low variability** → the model consistently agrees with the pseudo-label → high weight
- **Low confidence, high variability** → the model is unsure or inconsistent → low weight
- lambda1 (optimistic) gives benefit of the doubt; lambda2 (conservative) is more strict

```
        Confidence & Variability → Lambda Weights

   Sample A: conf=0.9, var=0.05      Sample B: conf=0.3, var=0.2
   ┌────────────────────────┐        ┌────────────────────────┐
   │ ████████████████████░  │ 0.9    │ ██████░░░░░░░░░░░░░░░  │ 0.3
   │ █░░░░░░░░░░░░░░░░░░░   │ 0.05   │ ████░░░░░░░░░░░░░░░░░  │ 0.2
   │                        │        │                        │
   │ lambda1 = 0.95         │        │ lambda1 = 0.50         │
   │ lambda2 = 0.85         │        │ lambda2 = 0.10         │
   │ → High trust ✓         │        │ → Low trust ✗          │
   └────────────────────────┘        └────────────────────────┘
```

### Phase 2 — Co-Training

**Goal**: Train strong classifiers on the large pseudo-labeled set, weighted by trust.

Two **new** BERT models are initialized fresh. They train on D_LG using **weighted cross-entropy loss**, where each sample's loss contribution is scaled by its lambda weight:

- **Model 1**'s loss is weighted by **lambda2** (conservative weights derived from Model 2's Phase 1 tracker)
- **Model 2**'s loss is weighted by **lambda1** (optimistic weights derived from Model 1's Phase 1 tracker)

This cross-weighting is the core of co-training — each model guides the other by sharing its perspective on pseudo-label quality. The weights are also updated each epoch as new probability observations accumulate.

```
    Cross-Weight Exchange in Co-Training

    Model 1                          Model 2
    ┌──────────────┐                ┌──────────────┐
    │              │   lambda2      │              │
    │  Trains on   │◄────────────── │  Provides    │
    │  D_LG with   │   (conservative│  conservative│
    │  lambda2 wts │    weights)    │  weights     │
    │              │                │              │
    │  Provides    │   lambda1      │  Trains on   │
    │  optimistic  │───────────────►│  D_LG with   │
    │  weights     │   (optimistic  │  lambda1 wts │
    │              │    weights)    │              │
    └──────────────┘                └──────────────┘
```

### Phase 3 — Fine-Tuning

**Goal**: Adapt the co-trained models to the clean labeled data.

Each co-trained model fine-tunes on its respective labeled split (Model 1 on D_l1, Model 2 on D_l2) with **early stopping** on the dev set (patience = 5 epochs). The stopping criterion is configurable via `--stopping-strategy` with six options: `baseline` (ensemble macro-F1, default), `no_early_stopping` (run all epochs), `per_class_patience` (stop only when every class plateaus), `weighted_macro_f1` (rare-class-weighted metric), `balanced_dev` (resampled dev set), and `scaled_threshold` (imbalance-scaled improvement threshold). This corrects any remaining bias from pseudo-label noise.

**Final evaluation** uses **ensemble prediction**: average the softmax probabilities from both models, then take the argmax.

```
    Ensemble Prediction

    Input tweet: "People trapped under rubble, need help!"

    Model 1 softmax:  [0.01, 0.02, 0.05, 0.60, 0.02, ...]
    Model 2 softmax:  [0.02, 0.01, 0.08, 0.55, 0.03, ...]
                       ─────────────────────────────────────
    Average:          [0.015, 0.015, 0.065, 0.575, 0.025, ...]
                                            ▲
                                         argmax
                                            │
    Prediction: "injured_or_dead_people" ◄──┘
```

---

## Class Labels

The full 10-class label set for humanitarian tweet classification (alphabetically sorted):

| Class                                    | Description                           | Example                                                          |
| ---------------------------------------- | ------------------------------------- | ---------------------------------------------------------------- |
| `caution_and_advice`                     | Warnings and safety advice            | _"Stay away from the coast, tsunami warning in effect"_          |
| `displaced_people_and_evacuations`       | Evacuations, displacement             | _"Thousands evacuated from flood zones"_                         |
| `infrastructure_and_utility_damage`      | Damage to buildings, roads, utilities | _"The bridge on Highway 1 has collapsed"_                        |
| `injured_or_dead_people`                 | Casualties and injuries               | _"At least 15 people confirmed dead"_                            |
| `missing_or_found_people`                | Missing/found persons                 | _"Has anyone seen my grandmother? She was in the affected area"_ |
| `not_humanitarian`                       | Irrelevant tweets                     | _"Just saw a great movie today"_                                 |
| `other_relevant_information`             | Other disaster-related info           | _"The earthquake was magnitude 7.8"_                             |
| `requests_or_urgent_needs`               | Calls for help and resources          | _"We need blankets and clean water urgently"_                    |
| `rescue_volunteering_or_donation_effort` | Rescue and aid efforts                | _"Red Cross volunteers arriving at the scene"_                   |
| `sympathy_and_support`                   | Expressions of sympathy               | _"Our thoughts and prayers go out to all affected"_              |

Not all events contain every class. The pipeline automatically detects the subset present in each event's data files using `detect_event_classes()`.

---

## Disaster Events

The dataset covers 10 real-world disaster events across 4 years:

| #   | Event                       | Type       | Year |
| --- | --------------------------- | ---------- | ---- |
| 1   | `california_wildfires_2018` | Wildfire   | 2018 |
| 2   | `canada_wildfires_2016`     | Wildfire   | 2016 |
| 3   | `cyclone_idai_2019`         | Cyclone    | 2019 |
| 4   | `hurricane_dorian_2019`     | Hurricane  | 2019 |
| 5   | `hurricane_florence_2018`   | Hurricane  | 2018 |
| 6   | `hurricane_harvey_2017`     | Hurricane  | 2017 |
| 7   | `hurricane_irma_2017`       | Hurricane  | 2017 |
| 8   | `hurricane_maria_2017`      | Hurricane  | 2017 |
| 9   | `kaikoura_earthquake_2016`  | Earthquake | 2016 |
| 10  | `kerala_floods_2018`        | Flood      | 2018 |

Each event has **4 budget levels** (5, 10, 25, 50 labeled tweets per class) and **3 seed sets**, giving **12 experiments per event** and **120 total experiments**.

---

## Data Layout

```
data/
├── original/
│   └── {event}/
│       ├── {event}_train.tsv                    # Full training set
│       ├── {event}_dev.tsv                      # Development set
│       ├── {event}_test.tsv                     # Test set
│       ├── labeled_{budget}_set{seed}.tsv       # Human-labeled subset (5/10/25/50 per class)
│       └── unlabeled_{budget}_set{seed}.tsv     # Remaining unlabeled tweets
│
└── pseudo-labelled/
    └── {source}/                                # e.g., "gpt-4o", "llama-3"
        └── {event}/
            └── {event}_train_pred.csv           # LLM pseudo-labels
```

**File formats**:

| File type          | Format          | Columns                                                   |
| ------------------ | --------------- | --------------------------------------------------------- |
| Original TSV files | Tab-separated   | `tweet_id`, `tweet_text`, `class_label`                   |
| Pseudo-label CSVs  | Comma-separated | `tweet_id`, `tweet_text`, `predicted_label`, `confidence` |

---

## Installation

```bash
pip install -r lg_cotrain/requirements.txt
```

**Dependencies**: `torch`, `transformers`, `pandas`, `scikit-learn`, `numpy`, `pytest`

---

## Usage

### Single Experiment

Run one experiment for a specific event, budget, and seed set:

```bash
python -m lg_cotrain.run_experiment \
    --event kaikoura_earthquake_2016 \
    --budget 5 \
    --seed-set 1
```

### Batch Mode (Multiple Events/Budgets/Seeds)

Run all 12 experiments (4 budgets x 3 seeds) for one event:

```bash
python -m lg_cotrain.run_experiment --event kaikoura_earthquake_2016
```

Run multiple events at once:

```bash
python -m lg_cotrain.run_experiment \
    --events california_wildfires_2018 canada_wildfires_2016 cyclone_idai_2019
```

Run specific budgets and seed sets:

```bash
python -m lg_cotrain.run_experiment \
    --event kaikoura_earthquake_2016 \
    --budgets 5 10 \
    --seed-sets 1 2
```

### Custom Pseudo-Label Source and Output Folder

Use a different pseudo-label source (e.g., from a different LLM) and store results in a named sub-folder:

```bash
python -m lg_cotrain.run_experiment \
    --events california_wildfires_2018 canada_wildfires_2016 \
    --pseudo-label-source llama-3 \
    --output-folder results/llama-3-run1
```

This reads pseudo-labels from `data/pseudo-labelled/llama-3/` and writes results to `results/llama-3-run1/`.

### All CLI Options

| Option                  | Description                         | Default                         |
| ----------------------- | ----------------------------------- | ------------------------------- |
| `--event`               | Single event name                   | _(required, or use `--events`)_ |
| `--events`              | One or more event names             | _(required, or use `--event`)_  |
| `--budget`              | Single budget value (5, 10, 25, 50) | All budgets                     |
| `--budgets`             | One or more budget values           | All budgets                     |
| `--seed-set`            | Single seed set (1, 2, 3)           | All seed sets                   |
| `--seed-sets`           | One or more seed sets               | All seed sets                   |
| `--pseudo-label-source` | Pseudo-label directory name         | `gpt-4o`                        |
| `--output-folder`       | Output folder for results           | `results/`                      |
| `--model-name`          | HuggingFace model name              | `bert-base-uncased`             |
| `--weight-gen-epochs`   | Phase 1 epochs                      | `7`                             |
| `--cotrain-epochs`      | Phase 2 epochs                      | `10`                            |
| `--finetune-max-epochs` | Phase 3 max epochs                  | `100`                           |
| `--finetune-patience`   | Early stopping patience             | `5`                             |
| `--stopping-strategy`  | Phase 3 early stopping strategy (`baseline`, `no_early_stopping`, `per_class_patience`, `weighted_macro_f1`, `balanced_dev`, `scaled_threshold`) | `baseline` |
| `--batch-size`          | Training batch size                 | `32`                            |
| `--lr`                  | Learning rate                       | `2e-5`                          |
| `--max-seq-length`      | Max token sequence length           | `128`                           |
| `--data-root`           | Path to data directory              | `data/`                         |
| `--results-root`        | Path to results directory           | `results/`                      |

### Interactive Notebooks

Three Jupyter notebooks are provided in the `Notebooks/` directory:

| Notebook                            | Description                                                                                                                                                                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_kaikoura_experiment.ipynb`      | Step-by-step walkthrough of the full pipeline on one event (Kaikoura Earthquake). Includes class distributions, per-epoch probability tracking, training curves, and per-class F1 charts.                                                                         |
| `02_all_disasters_experiment.ipynb` | Runs all 120 experiments (10 events x 4 budgets x 3 seeds) with resume support. Contains cross-disaster summary tables, line plots, and heatmaps.                                                                                                                 |
| `03_all_disasters_rerun.ipynb`      | Re-run all disasters with a **configurable pseudo-label source** and **named output folder**. Edit `PSEUDO_LABEL_SOURCE` and `RUN_NAME` in cell 2, then run all cells. Results are stored in `results/{RUN_NAME}/` to enable side-by-side comparison across runs. |
| `04_alternative_stopping_strategies.ipynb` | **Quick comparison** of all 6 stopping strategies (budget=50, seed=1, all 10 events = 60 runs). Results stored in `results/{pseudo_label_source}-quick-stop-{strategy}/`. Includes `ProgressTracker` for elapsed time and ETA. |
| `05_stopping_strategies_full_run.ipynb` | **Full sweep** across all stopping strategies (all budgets × seeds × events × strategies = 720 runs). Results stored in `results/{pseudo_label_source}-stop-{strategy}/`. Includes `ProgressTracker` for elapsed time and ETA. |

All notebooks support **resume** — if interrupted, they skip experiments that already have `metrics.json` files.

### Results Dashboard

Generate an HTML dashboard from experiment results:

```bash
# From the default results directory
python -m lg_cotrain.dashboard

# From a specific results directory
python -m lg_cotrain.dashboard --results-root results/gpt-4o-run1

# Multi-tab dashboard comparing multiple result sets
# (auto-detected when results/ contains sub-folders with metrics)
python -m lg_cotrain.dashboard --results-root results/
```

The dashboard is a self-contained HTML file with:

- Summary cards (total experiments, average F1, average error rate, average ECE)
- Pivot table grouped by budget showing mean/std across seeds per event
- All-results table with sorting by any column
- Lambda weight analysis table
- **Multi-tab view** when multiple result sets exist (e.g., different pseudo-label sources)

```
    Dashboard Layout (Multi-Tab Mode)

    ┌─────────────────────────────────────────────────┐
    │  [ gpt-4o-run1 ] [ llama-3-run1 ] [ gpt-4o-v2 ]│  ◄─ Tab bar
    ├─────────────────────────────────────────────────┤
    │                                                 │
    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
    │  │120 Exp │ │ 0.52   │ │ 34.2%  │ │ 0.082  │   │  ◄─ Summary cards
    │  │ Total  │ │Avg F1  │ │Avg Err │ │Avg ECE │   │
    │  └────────┘ └────────┘ └────────┘ └────────┘   │
    │                                                 │
    │  [ Pivot View ] [ All Results ]                 │  ◄─ View toggle
    │                                                 │
    │  ┌─────────────────────────────────────────┐    │
    │  │ Budget │ Seed 1    │ Seed 2    │ Seed 3 │    │
    │  │     5  │ 42.1 .481 │ 39.5 .503 │ ...    │    │  ◄─ Results table
    │  │    10  │ 38.2 .521 │ 36.1 .540 │ ...    │    │
    │  │    25  │ 33.5 .562 │ 32.0 .578 │ ...    │    │
    │  │    50  │ 29.1 .601 │ 28.3 .612 │ ...    │    │
    │  └─────────────────────────────────────────┘    │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

---

## Output Format

Results are saved to `results/{event}/{budget}_set{seed}/metrics.json` (or `results/{run_name}/{event}/{budget}_set{seed}/metrics.json` when using `--output-folder`):

```json
{
  "event": "kaikoura_earthquake_2016",
  "budget": 5,
  "seed_set": 1,
  "test_error_rate": 35.21,
  "test_macro_f1": 0.4812,
  "test_ece": 0.082,
  "test_per_class_f1": [0.52, 0.41, 0.38, ...],
  "dev_error_rate": 33.10,
  "dev_macro_f1": 0.5023,
  "dev_ece": 0.075,
  "stopping_strategy": "baseline",
  "lambda1_mean": 0.7234,
  "lambda1_std": 0.1456,
  "lambda2_mean": 0.5891,
  "lambda2_std": 0.1823
}
```

| Field                          | Description                            |
| ------------------------------ | -------------------------------------- |
| `test_error_rate`              | Error rate on held-out test set (%)    |
| `test_macro_f1`                | Macro-averaged F1 on test set          |
| `test_ece`                     | Expected Calibration Error on test set |
| `test_per_class_f1`            | Per-class F1 scores (list)             |
| `dev_error_rate`               | Error rate on development set (%)      |
| `dev_macro_f1`                 | Macro-averaged F1 on development set   |
| `dev_ece`                      | Expected Calibration Error on dev set  |
| `stopping_strategy`            | Phase 3 early stopping strategy used   |
| `lambda1_mean` / `lambda1_std` | Statistics of optimistic weights       |
| `lambda2_mean` / `lambda2_std` | Statistics of conservative weights     |

An experiment log is also saved to `experiment.log` in the same directory.

---

## Project Structure

```
lg_cotrain/                          # Main package
├── __init__.py
├── config.py                        # LGCoTrainConfig dataclass — auto-computes all file paths
├── data_loading.py                  # Data loading, label encoding, dataset splitting, TweetDataset
├── evaluate.py                      # Metrics (error rate, macro-F1, ECE, per-class F1), ensemble prediction
├── model.py                         # BertClassifier — wrapper around BertForSequenceClassification
├── trainer.py                       # LGCoTrainer — orchestrates the 3-phase pipeline
├── run_experiment.py                # CLI entry point (single + batch mode)
├── run_all.py                       # Batch runner: all budget x seed_set experiments for one event
├── dashboard.py                     # HTML dashboard generator (auto-discovery, multi-tab)
├── utils.py                         # Seed setting, logging, EarlyStopping + alternative stopping classes, device selection
├── weight_tracker.py                # Per-sample probability tracking and lambda weight computation
└── requirements.txt                 # Python dependencies

Notebooks/
├── 01_kaikoura_experiment.ipynb                 # Step-by-step pipeline walkthrough with visualizations
├── 02_all_disasters_experiment.ipynb            # Full 120-experiment run (preserved results)
├── 03_all_disasters_rerun.ipynb                 # Re-run with configurable pseudo-label source + output folder
├── 04_alternative_stopping_strategies.ipynb     # Quick comparison of stopping strategies (budget=50, seed=1)
└── 05_stopping_strategies_full_run.ipynb        # Full sweep across all strategies (720 runs)

tests/                               # 356 tests across 13 test files
├── conftest.py                      # Shared pytest fixtures
├── test_config.py                   # Config dataclass path computation and defaults (25 tests)
├── test_dashboard.py                # Dashboard HTML generation, auto-discovery, multi-tab (62 tests)
├── test_data_loading.py             # Data loading, label encoding, class detection (28 tests)
├── test_early_stopping.py           # PerClassEarlyStopping, EarlyStoppingWithDelta, class weight helpers (26 tests)
├── test_evaluate.py                 # Metric computation, ECE, ensemble (28 tests)
├── test_model.py                    # BertClassifier forward/predict_proba (4 tests)
├── test_notebook.py                 # Notebook 01 + 03 structure and content validation (56 tests)
├── test_notebook_02.py              # Notebook 02 structure validation (22 tests)
├── test_run_all.py                  # Batch runner, custom budgets/seeds/source (25 tests)
├── test_run_experiment.py           # CLI argument parsing and forwarding (12 tests)
├── test_trainer.py                  # Full pipeline integration test (4 tests)
├── test_utils.py                    # Seed, EarlyStopping, device (13 tests)
└── test_weight_tracker.py           # Lambda weight computation, seeding (31 tests)

docs/
└── Cornelia etal2025-Cotraining.pdf # Reference paper

data/                                # Disaster event datasets
results/                             # Experiment outputs + dashboard
```

### Module Dependency Graph

```
    run_experiment.py ──► run_all.py ──► trainer.py
                              │              │
                              │         ┌────┴─────────────────┐
                              │         │    │    │    │       │
                              │         ▼    ▼    ▼    ▼       ▼
                              │      config  data  model eval  utils
                              │              loading          weight
                              │                               tracker
                              ▼
                         dashboard.py ──► data_loading (CLASS_LABELS only)
```

---

## Testing

### Run the full test suite

```bash
python -m pytest tests/ -v
```

### Run a single test file

```bash
python -m pytest tests/test_weight_tracker.py -v
```

### Run a specific test

```bash
python -m pytest tests/test_trainer.py::TestFullPipelineTiny::test_full_pipeline -v
```

### Pure-Python tests (no torch/transformers required)

Several modules have pure-Python fallback paths and can be tested without ML dependencies:

```bash
python -m unittest tests/test_config.py
python -m unittest tests/test_weight_tracker.py
python -m unittest tests/test_evaluate.py
```

### Test Coverage Summary

| Test File                   | Tests | What It Covers                                                        |
| --------------------------- | ----- | --------------------------------------------------------------------- |
| `test_config.py`            | 25    | Path computation, defaults, pseudo-label source                       |
| `test_dashboard.py`         | 62    | HTML generation, event discovery, multi-tab, summary cards            |
| `test_data_loading.py`      | 28    | TSV/CSV loading, label encoding, class detection, D_LG building       |
| `test_early_stopping.py`    | 26    | PerClassEarlyStopping, EarlyStoppingWithDelta, class weight helpers   |
| `test_evaluate.py`          | 28    | Error rate, macro-F1, per-class F1, ECE, ensemble predict             |
| `test_model.py`             | 4     | BertClassifier forward pass, predict_proba                            |
| `test_notebook.py`          | 56    | Notebook 01 + 03 structure, imports, content, cell types              |
| `test_notebook_02.py`       | 22    | Notebook 02 structure and content                                     |
| `test_run_all.py`           | 25    | Batch runner, custom budgets/seeds, pseudo-label forwarding           |
| `test_run_experiment.py`    | 12    | CLI parsing, single/batch modes, argument forwarding                  |
| `test_trainer.py`           | 4     | Full 3-phase pipeline integration                                     |
| `test_utils.py`             | 13    | Seed reproducibility, EarlyStopping, device detection                 |
| `test_weight_tracker.py`    | 31    | Confidence, variability, lambda computation, tracker seeding          |

---

## Design Decisions

- **Lazy imports**: `data_loading.py` uses lazy imports for `torch`/`transformers`/`pandas` so that pure-Python modules (`config`, `weight_tracker`, `evaluate`) work without ML dependencies installed.

- **TweetDataset proxy**: `TweetDataset` is a lazy proxy class that only imports the real PyTorch `Dataset` on first instantiation, keeping the module importable without torch.

- **Pure-Python fallbacks**: `evaluate.py` and `weight_tracker.py` include fallback implementations that work without `numpy`/`scikit-learn`, enabling lightweight testing.

- **Dynamic class detection**: `detect_event_classes()` computes the union of classes across all data splits (labeled, unlabeled, dev, test) per event. This ensures no class appearing at test time is missed while avoiding unused output neurons.

- **Text cross-validation**: `build_d_lg()` verifies that tweet text matches between the unlabeled TSV and pseudo-label CSV when joining on `tweet_id`, logging warnings on mismatches.

- **Cross-platform paths**: `LGCoTrainConfig` computes all data and results paths using `pathlib.Path`, working on both Linux and Windows.

- **Configurable pseudo-label source**: The `pseudo_label_source` field in `LGCoTrainConfig` (default `"gpt-4o"`) determines which directory under `data/pseudo-labelled/` to read from, enabling experiments with different LLMs without code changes.

- **Result set sub-folders**: Results can be stored in named sub-folders (`results/gpt-4o-run1/`, `results/llama-3-run1/`) for side-by-side comparison. The dashboard auto-discovers all result sets.

- **Resume support**: Both `run_all_experiments()` and the notebooks skip experiments whose `metrics.json` already exists, making it safe to restart after crashes.

- **Dependency injection for testing**: `run_all_experiments()` accepts `_trainer_cls` and `_on_experiment_done` parameters, allowing tests to inject mock trainers without importing torch.

- **Configurable early stopping strategy**: The `stopping_strategy` field in `LGCoTrainConfig` (default `"baseline"`) selects how Phase 3 decides when to stop. Six strategies are supported: standard patience on ensemble macro-F1, no early stopping, per-class patience, rare-class-weighted metric, resampled dev set, and imbalance-scaled improvement threshold. Pass `--stopping-strategy` on the CLI to switch strategies without code changes.

- **Paper-aligned Phase 1 → Phase 2 seeding**: `WeightTracker.seed_from_last_epoch()` seeds Phase 2 with only the final Phase 1 epoch's probabilities, matching Algorithm 1 in the paper. The earlier `seed_from_tracker()` (full-history copy) is retained for reference.

---

## References

**Paper**:

> Md Mezbaur Rahman and Cornelia Caragea. 2025. **LLM-Guided Co-Training for Text Classification**. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)_, pages 31092–31109, Suzhou, China. Association for Computational Linguistics.

- arXiv: [https://arxiv.org/abs/2509.16516](https://arxiv.org/abs/2509.16516)
- ACL Anthology: [https://aclanthology.org/2025.emnlp-main.1583/](https://aclanthology.org/2025.emnlp-main.1583/)
- DOI: [https://doi.org/10.48550/arXiv.2509.16516](https://doi.org/10.48550/arXiv.2509.16516)
- Local copy: [`docs/Cornelia etal2025-Cotraining.pdf`](docs/Cornelia%20etal2025-Cotraining.pdf)
