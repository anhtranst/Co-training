# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LG-CoTrain is a semi-supervised co-training pipeline for humanitarian tweet classification during disaster events. It combines a small set of human-labeled tweets with GPT-4o pseudo-labeled tweets, using a 3-phase training approach with two BERT models. Based on the paper in `docs/Cornelia etal2025-Cotraining.pdf`.

## Commands

### Run an experiment
```bash
python -m lg_cotrain.run_experiment --event canada_wildfires_2016 --budget 5 --seed-set 1
```
Valid budgets: 5, 10, 25, 50. Valid seed sets: 1, 2, 3.

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run a single test file or test
```bash
python -m pytest tests/test_weight_tracker.py -v
python -m pytest tests/test_trainer.py::TestFullPipelineTiny::test_full_pipeline -v
```

### Run tests without torch/transformers (pure-Python subset)
```bash
python -m unittest tests/test_config.py
python -m unittest tests/test_weight_tracker.py
```

### Install dependencies
```bash
pip install -r lg_cotrain/requirements.txt
```

## Architecture

### 3-Phase Pipeline (`trainer.py` → `LGCoTrainer.run()`)

1. **Phase 1 — Weight Generation**: Two fresh BERT models are trained separately on D_l1 and D_l2 (stratified halves of the labeled set). After each epoch, both models' softmax probabilities over the pseudo-labeled set (D_LG) are recorded by `WeightTracker`.

2. **Phase 2 — Co-Training**: Two new BERT models train on D_LG using weighted cross-entropy. Model 1's loss is weighted by lambda2 (conservative weights from model 2's tracker), and model 2's loss is weighted by lambda1 (optimistic weights from model 1's tracker). Weights update each epoch.

3. **Phase 3 — Fine-Tuning**: Both co-trained models fine-tune on their respective labeled splits (D_l1, D_l2) with early stopping on dev macro-F1. Final evaluation uses ensemble prediction (averaged softmax → argmax).

### Lambda Weight Computation (`weight_tracker.py`)

- **Confidence** = mean of p(pseudo_label | x; θ) across recorded epochs
- **Variability** = std of the same
- **Lambda-optimistic** (λ1) = confidence + variability
- **Lambda-conservative** (λ2) = max(confidence - variability, 0)

### Data Layout

- `data/original/{event}/` — TSV files: `{event}_{train,dev,test}.tsv`, `labeled_{budget}_set{seed}.tsv`, `unlabeled_{budget}_set{seed}.tsv`
- `data/pseudo-labelled/gpt-4o/{event}/` — CSV: `{event}_train_pred.csv` with columns `tweet_id, tweet_text, predicted_label, confidence`
- 10 disaster events, 4 budget levels (5/10/25/50), 3 seed sets each
- Fixed 10-class label set defined in `data_loading.CLASS_LABELS` (alphabetically sorted). Not all events contain samples from every class

### Key Design Decisions

- `data_loading.py` uses lazy imports for torch/transformers/pandas so that pure-Python modules (config, weight_tracker, evaluate) work without ML dependencies
- `TweetDataset` is a lazy proxy class that imports the real PyTorch Dataset on first instantiation
- `evaluate.py` and `weight_tracker.py` have pure-Python fallback paths (no numpy/sklearn required)
- `LGCoTrainConfig` auto-computes all file paths in `__post_init__` from `event`, `budget`, `seed_set`, `data_root`, and `results_root`
- Results are saved to `results/{event}/{budget}_set{seed}/metrics.json`

### Paper vs Implementation Deviations

Documented differences between Algorithm 1 in the paper (`docs/Cornelia etal2025-Cotraining.pdf`) and this implementation:

- **Per-epoch lambda updates**: The paper's pseudocode (Algorithm 1, lines 24-28) shows updating confidence, variability, and lambdas for each mini-batch during Phase 2 co-training. This implementation updates lambdas once per epoch via a full evaluation pass over D_LG (`trainer.py` lines 261-267). Per-epoch updates are more computationally stable and are standard practice in semi-supervised learning.

- **Lambda-conservative clipping**: The paper's Eq. 4 defines `λ2_i = c_θ2 - v_θ2` without explicit clipping. This implementation clips to `max(c - v, 0)` (`weight_tracker.py` `compute_lambda_conservative`). Negative weights would invert the loss gradient, encouraging the model toward incorrect predictions. Clipping to zero effectively excludes low-confidence, high-variability samples.

- **Early stopping criterion**: The paper specifies patience=5 on dev macro-F1 during Phase 3 fine-tuning but does not detail whether evaluation is per-model or ensemble-based. This implementation uses the ensemble macro-F1 (averaged softmax of both models) as the stopping metric for both models' early stopping trackers. Both models must independently exhaust patience before training stops (`trainer.py` lines 316-325). Using ensemble F1 is reasonable because the ensemble is the final evaluation artifact.

## Workflow Rules

- Every code change (new or modified) must include corresponding test cases in `tests/`. After writing tests, run `python -m pytest tests/ -v` to verify all tests pass before considering the change complete.
