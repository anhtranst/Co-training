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

Two fresh BERT models are trained **separately** — Model 1 on D_l1, Model 2 on D_l2 (stratified halves of the small labeled set). After training completes, each model predicts softmax probabilities for every sample in the pseudo-labeled set (D_LG) using only the **final epoch's** weights. These final-epoch probabilities are then used to seed Phase 2's `WeightTracker` (via `seed_from_last_epoch()`), matching Algorithm 1 in the paper.

Since Phase 2 starts with only one epoch of observations, the initial lambda weights are:

```
confidence  = p(pseudo_label | x; theta) from the final Phase 1 epoch
variability = 0  (only one observation)

lambda_optimistic (lambda1)    = confidence + 0 = confidence
lambda_conservative (lambda2)  = max(confidence - 0, 0) = confidence
```

As Phase 2 training proceeds and new probability observations accumulate each epoch, confidence and variability are recomputed as the mean and std across all recorded Phase 2 epochs, causing the optimistic and conservative weights to diverge.

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

- **Model 1**'s loss is weighted by **lambda2** (conservative weights seeded from Model 2's final Phase 1 epoch)
- **Model 2**'s loss is weighted by **lambda1** (optimistic weights seeded from Model 1's final Phase 1 epoch)

This cross-weighting is the core of co-training — each model guides the other by sharing its perspective on pseudo-label quality. Each Phase 2 epoch, both models re-evaluate D_LG and the new probability observations are added to the tracker, so confidence and variability (and thus lambda weights) are recomputed with increasing history.

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

Each co-trained model fine-tunes on its respective labeled split (Model 1 on D_l1, Model 2 on D_l2) with **early stopping** on the dev set (patience = 5 epochs, configurable). This corrects any remaining bias from pseudo-label noise.

**Final evaluation** uses **ensemble prediction**: average the softmax probabilities from both models, then take the argmax.

#### Early Stopping Strategies

The stopping criterion is selected with `--stopping-strategy`. All six strategies restore the best-ever model checkpoint when training ends, so the final model is never the last epoch but always the peak-performance epoch.

The **core problem** that alternatives address: on imbalanced disaster datasets, majority classes (e.g. `not_humanitarian`) converge quickly and plateau macro-F1, causing `baseline` to stop before rare classes (e.g. `missing_or_found_people`) have finished learning.

| Strategy | How it decides to stop | Best used when |
| --- | --- | --- |
| `baseline` | Stops when **ensemble macro-F1** on the full dev set has not improved for `patience` epochs. Both models must independently exhaust patience. | Balanced class distributions; a good starting point for any event. |
| `no_early_stopping` | Runs **all `finetune_max_epochs`** (default 100) and restores the best checkpoint seen across all epochs. Never stops early. | Diagnosing whether `baseline` stopped too soon; provides an upper-bound reference for other strategies. |
| `per_class_patience` | Tracks F1 for **each class independently**. Stops only when **every** class has individually plateaued (patience exhausted per class). Checkpoints on improvement of the mean per-class F1. | Highly imbalanced events where rare classes are still improving long after majority classes plateau. |
| `weighted_macro_f1` | Computes a **rare-class-weighted** stopping metric: each class's F1 is multiplied by its inverse frequency (normalized to mean weight = 1.0), so rare classes contribute proportionally more to the stopping signal. | Events with a few dominant classes and several rare ones — rare-class improvements are not washed out by majority-class plateaus. |
| `balanced_dev` | Resamples the dev set to **equal class counts** (down-sampled to the smallest class) before computing the stopping metric. The full dev set is still used for logging. | Events with very large majority classes that dominate raw macro-F1; equal representation makes the stopping signal sensitive to all classes equally. |
| `scaled_threshold` | Requires a **minimum improvement delta** before resetting patience. The delta scales with the class imbalance ratio (`max_freq / min_freq`): a ratio of 1 gives `delta = 0.001`; a ratio of 10 gives `delta = 0.01` (capped at 20×). | Highly imbalanced events where tiny noise fluctuations in macro-F1 can spuriously reset patience and delay stopping. |

```
    Which strategy fires earliest vs latest (typical ordering):

    baseline  ──────────────────────►  stops ~patience epochs after F1 peaks
    weighted_macro_f1 ───────────────►  similar to baseline but shifts weight to rare classes
    balanced_dev ────────────────────►  similar, resampling equalises class influence
    scaled_threshold ────────────────►  stops later when fluctuations stay below min_delta
    per_class_patience ──────────────►  stops latest: waits for the slowest class
    no_early_stopping ───────────────►  never stops early (runs all epochs)
```

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

**Dependencies**: `torch`, `transformers`, `pandas`, `scikit-learn`, `numpy`, `optuna`, `pytest`

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

### Hyperparameter Tuning with Optuna

#### Global tuning (one set of hyperparameters for all experiments)

Find optimal hyperparameters using a global Optuna study. Each trial runs the full 3-phase pipeline across all 10 events (budget=50, seed=1 by default), optimizing mean dev macro-F1:

```bash
# Run 20 trials across all 10 events
python -m lg_cotrain.optuna_tuner --n-trials 20

# Tune on a subset of events
python -m lg_cotrain.optuna_tuner --n-trials 10 --events hurricane_harvey_2017 kerala_floods_2018

# Use SQLite storage for resumable studies
python -m lg_cotrain.optuna_tuner --n-trials 20 --storage sqlite:///optuna.db
```

The tuner searches over: `lr` (1e-5 to 1e-3), `batch_size` ([8, 16, 32, 64]), `cotrain_epochs` (5-20), and `finetune_patience` (4-10). After the study completes, apply the best parameters to your experiments:

```bash
python -m lg_cotrain.run_experiment \
    --events california_wildfires_2018 canada_wildfires_2016 \
    --lr 3.5e-4 --batch-size 16 --cotrain-epochs 12 --finetune-patience 7
```

#### Per-experiment tuning (120 separate studies)

Find experiment-specific optimal hyperparameters — one Optuna study per (event, budget, seed_set) combination. Each study optimizes `dev_macro_f1` over 6 hyperparameters (lr, batch_size, cotrain_epochs, finetune_patience, weight_decay, warmup_ratio). Studies run in parallel across GPUs and support **incremental scaling**:

```bash
# Run all 120 studies with 10 trials each on 2 GPUs
python -m lg_cotrain.optuna_per_experiment --n-trials 10 --num-gpus 2

# Later, scale to 20 trials (continues from 10, only runs 10 new trials per study)
python -m lg_cotrain.optuna_per_experiment --n-trials 20 --num-gpus 2

# Tune a subset
python -m lg_cotrain.optuna_per_experiment --n-trials 10 \
    --events hurricane_harvey_2017 --budgets 50 --seed-sets 1
```

Results are saved under `results/optuna/per_experiment/{event}/{budget}_set{seed}/trials_{n}/best_params.json`. Each trial count gets its own subfolder — previous results are never overwritten. Use `load_best_params()` to load the latest results, or `load_best_params(n_trials=10)` for a specific trial count.

#### Monitoring progress

While Optuna studies are running, use the standalone progress checker to see trial-level progress and ETA:

```bash
# One-time snapshot
python check_progress.py

# Auto-refresh every 30 seconds
python check_progress.py --watch

# Custom refresh interval
python check_progress.py --watch --interval 10

# Custom results directory
python check_progress.py --results-dir /path/to/results
```

The script scans `study.log` files across all 120 studies and reports: completed/running/pending studies, total trial progress with a progress bar, average trial duration, estimated time remaining (accounting for GPU parallelism), and per-study detail showing current phase and epoch. It is read-only and does not modify any files.

#### Multi-PC workflow

To speed up tuning, split events across multiple PCs. On each PC, set `EVENTS` in notebook cell 3 to a subset (e.g., PC 1 runs 4 events, PC 2 runs 3, PC 3 runs 3). After all PCs finish, copy the result folders together and regenerate the summary:

```bash
# Regenerate summary from whatever exists (after manually copying folders)
python merge_optuna_results.py --target results/optuna/per_experiment --n-trials 10

# Or merge from other PCs and regenerate in one step
python merge_optuna_results.py \
    --sources pc2_results/optuna/per_experiment \
              pc3_results/optuna/per_experiment \
    --target results/optuna/per_experiment \
    --n-trials 10

# Preview what would be copied (no actual changes)
python merge_optuna_results.py \
    --sources pc2_results/optuna/per_experiment \
    --target results/optuna/per_experiment \
    --n-trials 10 --dry-run
```

The script reports completed, missing, and failed studies, and writes `summary_{n}.json` compatible with notebook cell 7.

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
| `--weight-decay`        | AdamW weight decay                  | `0.01`                          |
| `--warmup-ratio`        | LR scheduler warmup ratio           | `0.1`                           |
| `--max-seq-length`      | Max token sequence length           | `128`                           |
| `--num-gpus`            | Number of GPUs for parallel execution | `1` (sequential)              |
| `--data-root`           | Path to data directory              | `data/`                         |
| `--results-root`        | Path to results directory           | `results/`                      |

### Interactive Notebooks

Eight Jupyter notebooks are provided in the `Notebooks/` directory:

| Notebook                            | Description                                                                                                                                                                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `00_optuna_hyperparameter_tuning.ipynb` | **Global Optuna hyperparameter tuning**. Runs a Bayesian search over lr, batch_size, cotrain_epochs, and finetune_patience across all 10 events (budget=50, seed=1). Optimizes mean dev macro-F1. Includes optimization history plots, parameter distributions, and CLI commands to apply the best params. |
| `01_kaikoura_experiment.ipynb`      | Step-by-step walkthrough of the full pipeline on one event (Kaikoura Earthquake). Includes class distributions, per-epoch probability tracking, training curves, and per-class F1 charts.                                                                         |
| `02_all_disasters_experiment.ipynb` | Runs all 120 experiments (10 events x 4 budgets x 3 seeds) with resume support. Contains cross-disaster summary tables, line plots, and heatmaps.                                                                                                                 |
| `03_all_disasters_rerun.ipynb`      | Re-run all disasters with a **configurable pseudo-label source** and **named output folder**. Edit `PSEUDO_LABEL_SOURCE` and `RUN_NAME` in cell 2, then run all cells. Results are stored in `results/{source}/test/{run_name}/` to enable side-by-side comparison across runs. |
| `04_alternative_stopping_strategies.ipynb` | **Quick comparison** of all 6 stopping strategies (budget=50, seed=1, all 10 events = 60 runs). Results stored in `results/{source}/quick-stop/{strategy}/`. Includes `ProgressTracker` for elapsed time and ETA. |
| `05_stopping_strategies_full_run.ipynb` | **Full sweep** across all stopping strategies (all budgets × seeds × events × strategies = 720 runs). Results stored in `results/{source}/stop/{strategy}/`. Includes `ProgressTracker` for elapsed time and ETA. |
| `06_all_disasters_adamw_run.ipynb` | **Full 120-experiment run with AdamW** optimizer, linear LR scheduler, and 10% warmup (run-3). Uses baseline stopping strategy, paper-default hyperparameters, and **multi-GPU parallel execution** (`NUM_GPUS=2`). Results stored in `results/gpt-4o/test/run-3/`. |
| `07_optuna_per_experiment.ipynb` | **Per-experiment Optuna tuning**: 120 separate studies (one per event/budget/seed), each optimizing 6 hyperparameters (lr, batch_size, cotrain_epochs, finetune_patience, weight_decay, warmup_ratio) over N trials. Multi-GPU parallel execution. **Incremental scaling**: results stored under `trials_{n}/` subfolders; running with more trials continues from previous runs. |

All notebooks support **resume** — if interrupted, they skip experiments that already have `metrics.json` files.

### Results Dashboard

Generate an HTML dashboard from experiment results:

```bash
# From the default results directory (auto-discovers 3-level hierarchy)
python -m lg_cotrain.dashboard

# From a specific results directory
python -m lg_cotrain.dashboard --results-root results/

# Nested tab dashboard: model → type → experiment
# (auto-detected from results/{model}/{type}/{experiment}/ structure)
python -m lg_cotrain.dashboard --results-root results/
```

The dashboard is a self-contained HTML file with:

- Summary cards (total experiments, average F1, average error rate, average ECE)
- Pivot table grouped by budget showing mean/std across seeds per event
- All-results table with sorting by any column
- Lambda weight analysis table
- **3-level nested tab view** when multiple result sets exist — Level 1 (model), Level 2 (experiment type), Level 3 (experiment name)
- **Optuna tab** (appears between Data Analysis and model tabs when `results/optuna/optuna_results.json` exists) — shows best hyperparameters, search space, and all trial results

```
    Dashboard Layout (Multi-Tab Mode — 3-Level Nested Tabs)

    ┌───────────────────────────────────────────────────────┐
    │  [Data Analysis] [Optuna] [ gpt-4o ] [ llama-3 ]     │  ◄─ Level 1 (model)
    ├───────────────────────────────────────────────────────┤
    │  [ quick-stop ] [ stop ] [ test ]                     │  ◄─ Level 2 (type)
    ├───────────────────────────────────────────────────────┤
    │  [baseline] [no_early_stopping] [per_class_patience]  │  ◄─ Level 3 (experiment)
    ├───────────────────────────────────────────────────────┤
    │                                                       │
    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
    │  │120 Exp │ │ 0.52   │ │ 34.2%  │ │ 0.082  │         │  ◄─ Summary cards
    │  │ Total  │ │Avg F1  │ │Avg Err │ │Avg ECE │         │
    │  └────────┘ └────────┘ └────────┘ └────────┘         │
    │                                                       │
    │  [ Pivot View ] [ All Results ]                       │  ◄─ View toggle
    │                                                       │
    │  ┌─────────────────────────────────────────┐          │
    │  │ Budget │ Seed 1    │ Seed 2    │ Seed 3 │          │
    │  │     5  │ 42.1 .481 │ 39.5 .503 │ ...    │          │  ◄─ Results table
    │  │    10  │ 38.2 .521 │ 36.1 .540 │ ...    │          │
    │  │    25  │ 33.5 .562 │ 32.0 .578 │ ...    │          │
    │  │    50  │ 29.1 .601 │ 28.3 .612 │ ...    │          │
    │  └─────────────────────────────────────────┘          │
    │                                                       │
    └───────────────────────────────────────────────────────┘
```

---

## Output Format

Results are saved under a 3-level hierarchy: `results/{model}/{type}/{experiment}/{event}/{budget}_set{seed}/metrics.json` (e.g., `results/gpt-4o/quick-stop/baseline/canada_wildfires_2016/5_set1/metrics.json`):

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
├── parallel.py                      # Multi-GPU parallel execution (ProcessPoolExecutor with spawn)
├── dashboard.py                     # HTML dashboard generator (3-level nested tabs, auto-discovery)
├── optuna_tuner.py                  # Global Optuna hyperparameter tuner (standalone, uses dev macro-F1)
├── optuna_per_experiment.py         # Per-experiment Optuna tuner (120 studies, multi-GPU, JSON results)
├── utils.py                         # Seed setting, logging, EarlyStopping + alternative stopping classes, device selection
├── weight_tracker.py                # Per-sample probability tracking and lambda weight computation
└── requirements.txt                 # Python dependencies

Notebooks/
├── 00_optuna_hyperparameter_tuning.ipynb       # Global Optuna hyperparameter tuning (mean dev F1 across events)
├── 01_kaikoura_experiment.ipynb                 # Step-by-step pipeline walkthrough with visualizations
├── 02_all_disasters_experiment.ipynb            # Full 120-experiment run (preserved results)
├── 03_all_disasters_rerun.ipynb                 # Re-run with configurable pseudo-label source + output folder
├── 04_alternative_stopping_strategies.ipynb     # Quick comparison of stopping strategies (budget=50, seed=1)
├── 05_stopping_strategies_full_run.ipynb        # Full sweep across all strategies (720 runs)
├── 06_all_disasters_adamw_run.ipynb             # AdamW + linear scheduler run-3 (multi-GPU parallel)
└── 07_optuna_per_experiment.ipynb               # Per-experiment Optuna tuning (120 studies, multi-GPU)

check_progress.py                    # Standalone Optuna progress checker (study.log scanner with ETA)
merge_optuna_results.py              # Standalone Optuna results merger and summary generator

tests/                               # 400+ tests across 16 test files
├── conftest.py                      # Shared pytest fixtures
├── test_config.py                   # Config dataclass path computation and defaults (28 tests)
├── test_dashboard.py                # Dashboard HTML generation, auto-discovery, multi-tab, Optuna tab (105 tests)
├── test_data_loading.py             # Data loading, label encoding, class detection (28 tests)
├── test_early_stopping.py           # PerClassEarlyStopping, EarlyStoppingWithDelta, class weight helpers (26 tests)
├── test_evaluate.py                 # Metric computation, ECE, ensemble (28 tests)
├── test_model.py                    # BertClassifier forward/predict_proba (4 tests)
├── test_notebook.py                 # Notebook 01 + 03 structure and content validation (56 tests)
├── test_notebook_02.py              # Notebook 02 structure validation (22 tests)
├── test_optuna_per_experiment.py     # Per-experiment Optuna tuner, incremental trials, resume, GPU assignment, load_best_params (31 tests)
├── test_parallel.py                 # Multi-GPU parallel dispatch, resume, round-robin, callbacks (11 tests)
├── test_run_all.py                  # Batch runner, custom budgets/seeds/source (25 tests)
├── test_run_experiment.py           # CLI argument parsing and forwarding (12 tests)
├── test_optuna_tuner.py             # Global Optuna tuner, pruning, CLI parsing (20 tests)
├── test_trainer.py                  # Full pipeline integration test (4 tests)
├── test_utils.py                    # Seed, EarlyStopping, device (13 tests)
└── test_weight_tracker.py           # Lambda weight computation, seeding (31 tests)

docs/
└── Cornelia etal2025-Cotraining.pdf # Reference paper

data/                                # Disaster event datasets
results/                             # Experiment outputs + dashboard
├── dashboard.html                   # Auto-generated HTML dashboard
└── {model}/                         # e.g., gpt-4o
    ├── quick-stop/                  # Quick stopping strategy experiments
    │   └── {strategy}/              # e.g., baseline, per_class_patience
    │       └── {event}/{budget}_set{seed}/metrics.json
    ├── stop/                        # Full stopping strategy experiments
    │   └── {strategy}/...
    └── test/                        # General test runs
        └── {run_name}/...           # e.g., run-1, run-2
```

### Module Dependency Graph

```
    optuna_tuner.py ─────────► trainer.py (via _trainer_cls injection)
    optuna_per_experiment.py ─► trainer.py (via _trainer_cls injection)
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
| `test_dashboard.py`         | 105   | HTML generation, event discovery, 3-level nested tabs, Optuna tab     |
| `test_data_loading.py`      | 28    | TSV/CSV loading, label encoding, class detection, D_LG building       |
| `test_early_stopping.py`    | 26    | PerClassEarlyStopping, EarlyStoppingWithDelta, class weight helpers   |
| `test_evaluate.py`          | 28    | Error rate, macro-F1, per-class F1, ECE, ensemble predict             |
| `test_model.py`             | 4     | BertClassifier forward pass, predict_proba                            |
| `test_notebook.py`          | 56    | Notebook 01 + 03 structure, imports, content, cell types              |
| `test_notebook_02.py`       | 22    | Notebook 02 structure and content                                     |
| `test_optuna_tuner.py`      | 20    | Global Optuna study, objective function, pruning, CLI args            |
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

- **3-level results hierarchy**: Results are organized in a 3-level folder structure (`results/{model}/{type}/{experiment}/`) instead of flat names. For example, `results/gpt-4o/quick-stop/baseline/` instead of `results/gpt-4o-quick-stop-baseline/`. The dashboard uses nested tab bars (model → type → experiment) to navigate without horizontal overflow.

- **Optuna tuners**: Two self-contained Optuna tuning modes are available. `optuna_tuner.py` runs a global study (mean dev macro-F1 across all events) to find one set of hyperparameters. `optuna_per_experiment.py` runs 120 separate studies (one per event/budget/seed) to find experiment-specific optimal hyperparameters over 6 dimensions. Both use dev macro-F1 as the objective (no test-set leakage). Per-experiment studies run in parallel across GPUs, save results as JSON under `trials_{n}/` subfolders, and support incremental scaling — running with more trials continues from previous runs (via `study.add_trial()` replay).

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
