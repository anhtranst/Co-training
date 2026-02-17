# LG-CoTrain

> **Results Dashboard** — View experiment results: [results/dashboard.html](https://htmlpreview.github.io/?https://github.com/anhtranst/Co-training/blob/main/results/dashboard.html)
> _(Open locally in a browser; rebuild anytime with `python results/build_dashboard.py`)_

A semi-supervised co-training pipeline for humanitarian tweet classification during disaster events. Combines a small set of human-labeled tweets with GPT-4o pseudo-labeled tweets using a 3-phase training approach with two BERT models.

Based on the paper: _Cornelia et al. (2025) — Co-Training with LLM-Generated Pseudo-Labels_ (see `docs/Cornelia etal2025-Cotraining.pdf`).

## Overview

During disasters, rapid classification of social media posts helps humanitarian organizations prioritize response efforts. LG-CoTrain addresses the challenge of limited labeled data by:

1. Using GPT-4o to generate pseudo-labels for unlabeled tweets
2. Computing per-sample reliability weights via a dual-model weight generation scheme
3. Co-training two BERT classifiers with weighted cross-entropy loss
4. Fine-tuning on the small labeled set with early stopping

The pipeline dynamically detects which classes are present in each event's data (typically 8-10 of the 10 possible classes) and sizes the models accordingly.

## Class Labels

The full 10-class superset for humanitarian tweet classification:

| Class                                    | Description                           |
| ---------------------------------------- | ------------------------------------- |
| `caution_and_advice`                     | Warnings and safety advice            |
| `displaced_people_and_evacuations`       | Evacuations, displacement             |
| `infrastructure_and_utility_damage`      | Damage to buildings, roads, utilities |
| `injured_or_dead_people`                 | Casualties and injuries               |
| `missing_or_found_people`                | Missing/found persons                 |
| `not_humanitarian`                       | Irrelevant tweets                     |
| `other_relevant_information`             | Other disaster-related info           |
| `requests_or_urgent_needs`               | Calls for help and resources          |
| `rescue_volunteering_or_donation_effort` | Rescue and aid efforts                |
| `sympathy_and_support`                   | Expressions of sympathy               |

Not all events contain every class. The pipeline automatically detects the subset present in each event's data files.

## 3-Phase Pipeline

### Phase 1: Weight Generation

Two fresh BERT models train separately on stratified halves (D_l1 and D_l2) of the labeled set. After each epoch, both models' softmax probabilities over the pseudo-labeled set (D_LG) are recorded by a `WeightTracker`. At the end, lambda weights are computed:

- **Lambda-optimistic** (lambda1) = confidence + variability
- **Lambda-conservative** (lambda2) = max(confidence - variability, 0)

Where confidence = mean and variability = std of p(pseudo_label | x; theta) across epochs.

### Phase 2: Co-Training

Two new BERT models train on D_LG using weighted cross-entropy. Model 1's loss is weighted by lambda2 (conservative weights from model 2's tracker), and model 2's loss is weighted by lambda1 (optimistic weights from model 1's tracker). Weights are recomputed each epoch as new probability observations are recorded.

### Phase 3: Fine-Tuning

Both co-trained models fine-tune on their respective labeled splits (D_l1, D_l2) with early stopping on dev set macro-F1 (patience=5). Final evaluation uses ensemble prediction: averaged softmax probabilities from both models, then argmax.

## Installation

```bash
pip install -r lg_cotrain/requirements.txt
```

Dependencies: `torch`, `transformers`, `pandas`, `scikit-learn`, `numpy`, `pytest`

## Data Layout

```
data/
  original/{event}/
    {event}_train.tsv             # Full training set
    {event}_dev.tsv               # Dev set
    {event}_test.tsv              # Test set
    labeled_{budget}_set{seed}.tsv    # Human-labeled subset
    unlabeled_{budget}_set{seed}.tsv  # Remaining unlabeled tweets
  pseudo-labelled/gpt-4o/{event}/
    {event}_train_pred.csv        # GPT-4o pseudo-labels
```

TSV files have columns: `tweet_id`, `tweet_text`, `class_label`.
Pseudo-label CSVs have columns: `tweet_id`, `tweet_text`, `predicted_label`, `confidence`.

### Disaster Events

| Event                     | Year |
| ------------------------- | ---- |
| california_wildfires_2018 | 2018 |
| canada_wildfires_2016     | 2016 |
| cyclone_idai_2019         | 2019 |
| hurricane_dorian_2019     | 2019 |
| hurricane_florence_2018   | 2018 |
| hurricane_harvey_2017     | 2017 |
| hurricane_irma_2017       | 2017 |
| hurricane_maria_2017      | 2017 |
| kaikoura_earthquake_2016  | 2016 |
| kerala_floods_2018        | 2018 |

Each event has 4 budget levels (5, 10, 25, 50 labeled tweets per class) and 3 seed sets.

## Usage

### Run an experiment (CLI)

```bash
python -m lg_cotrain.run_experiment --event kaikoura_earthquake_2016 --budget 5 --seed-set 1
```

All CLI options:

```
--event              Event name (default: canada_wildfires_2016)
--budget             Label budget per class: 5, 10, 25, or 50
--seed-set           Seed set: 1, 2, or 3
--model-name         HuggingFace model (default: bert-base-uncased)
--weight-gen-epochs  Phase 1 epochs (default: 7)
--cotrain-epochs     Phase 2 epochs (default: 10)
--finetune-max-epochs  Phase 3 max epochs (default: 100)
--finetune-patience  Early stopping patience (default: 5)
--batch-size         Batch size (default: 32)
--lr                 Learning rate (default: 2e-5)
--max-seq-length     Max token length (default: 128)
--data-root          Path to data directory
--results-root       Path to results directory
```

### Run via notebook

Open `Notebooks/01_kaikoura_experiment.ipynb` for a step-by-step walkthrough of the pipeline with visualizations. The notebook includes:

- Class distribution analysis
- Per-epoch probability tracking plots
- Training loss curves
- Dev set F1 progression
- Per-class F1 bar charts
- Results export to JSON

### Output

Results are saved to `results/{event}/{budget}_set{seed}/metrics.json`:

```json
{
  "event": "kaikoura_earthquake_2016",
  "budget": 5,
  "seed_set": 1,
  "test_error_rate": 35.21,
  "test_macro_f1": 0.4812,
  "test_per_class_f1": [0.52, 0.41, ...],
  "dev_error_rate": 33.10,
  "dev_macro_f1": 0.5023,
  "lambda1_mean": 0.7234,
  "lambda1_std": 0.1456,
  "lambda2_mean": 0.5891,
  "lambda2_std": 0.1823
}
```

An experiment log is also saved to `results/{event}/{budget}_set{seed}/experiment.log`.

## Project Structure

```
lg_cotrain/
  __init__.py
  config.py           # LGCoTrainConfig dataclass with auto-computed paths
  data_loading.py     # Data loading, label encoding, dataset splitting, TweetDataset
  evaluate.py         # Metrics (error rate, macro-F1, per-class F1) and ensemble prediction
  model.py            # BertClassifier wrapper around BertForSequenceClassification
  run_experiment.py   # CLI entry point
  trainer.py          # LGCoTrainer: orchestrates the 3-phase pipeline
  utils.py            # Seed setting, logging, EarlyStopping, device selection
  weight_tracker.py   # Per-sample probability tracking and lambda weight computation
  requirements.txt    # Python dependencies

Notebooks/
  01_kaikoura_experiment.ipynb  # Interactive experiment walkthrough

tests/
  conftest.py              # Shared pytest fixtures
  test_config.py           # Config dataclass tests
  test_data_loading.py     # Data loading, label encoding, class detection tests
  test_evaluate.py         # Metric computation tests
  test_model.py            # BertClassifier tests
  test_notebook.py         # Notebook structure and content validation
  test_trainer.py          # Full pipeline integration test
  test_utils.py            # Seed, EarlyStopping, device tests
  test_weight_tracker.py   # WeightTracker computation tests

data/                      # Disaster event datasets (not committed)
docs/                      # Reference paper
results/                   # Experiment outputs (regenerated by experiments)
```

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run a single test file:

```bash
python -m pytest tests/test_weight_tracker.py -v
```

Run a specific test:

```bash
python -m pytest tests/test_trainer.py::TestFullPipelineTiny::test_full_pipeline -v
```

Some modules (config, weight_tracker, evaluate) have pure-Python fallback paths and can be tested without ML dependencies:

```bash
python -m unittest tests/test_config.py
python -m unittest tests/test_weight_tracker.py
```

## Design Decisions

- **Lazy imports**: `data_loading.py` uses lazy imports for torch/transformers/pandas so that pure-Python modules (config, weight_tracker, evaluate) work without ML dependencies installed.
- **Dynamic class detection**: `detect_event_classes()` computes the union of classes across all data splits (labeled, unlabeled, dev, test) per event. This ensures no class appearing at test time is missed, while avoiding unused output neurons.
- **Text cross-validation**: `build_d_lg()` verifies that tweet text matches between the unlabeled TSV and pseudo-label CSV when joining on `tweet_id`, logging warnings on mismatches.
- **Pure-Python fallbacks**: `evaluate.py` and `weight_tracker.py` include fallback implementations that work without numpy/sklearn, enabling lightweight testing.
- **Cross-platform paths**: `LGCoTrainConfig` computes data and results paths relative to the package location using `pathlib.Path`, working on both Linux and Windows.
