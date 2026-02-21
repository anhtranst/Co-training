"""Per-experiment Optuna hyperparameter tuning for LG-CoTrain.

Runs 120 separate Optuna studies — one for each (event, budget, seed_set)
combination — to find experiment-specific optimal hyperparameters.  Studies
run in parallel across multiple GPUs using ProcessPoolExecutor.

Results are saved as JSON files (no database).  Resume is handled by checking
for existing ``best_params.json`` files.

Usage::

    python -m lg_cotrain.optuna_per_experiment --n-trials 15 --num-gpus 2
    python -m lg_cotrain.optuna_per_experiment --n-trials 10 --events hurricane_harvey_2017
    python -m lg_cotrain.optuna_per_experiment --n-trials 15 --budget 50 --seed-set 1
"""

import argparse
import json
import logging
import multiprocessing as mp
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("lg_cotrain")

ALL_EVENTS = [
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

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]


def create_per_experiment_objective(
    event: str,
    budget: int,
    seed_set: int,
    device: Optional[str] = None,
    data_root: str = "/workspace/data",
    pseudo_label_source: str = "gpt-4o",
    _trainer_cls=None,
):
    """Return an Optuna objective function for a single experiment.

    Each call to the returned function:

    1. Samples 6 hyperparameters from the Optuna trial.
    2. Runs the full 3-phase pipeline for the given experiment.
    3. Returns ``dev_macro_f1`` (no test-set leakage).

    Parameters
    ----------
    event, budget, seed_set : experiment identifiers
    device : GPU device string (e.g. "cuda:0") or None for auto-detect
    data_root : base directory for input data
    pseudo_label_source : pseudo-label directory name
    _trainer_cls : override trainer class (for testing with mocks)
    """

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        cotrain_epochs = trial.suggest_int("cotrain_epochs", 5, 20)
        finetune_patience = trial.suggest_int("finetune_patience", 4, 10)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)

        # Use a temp directory so trial outputs don't pollute real results
        with tempfile.TemporaryDirectory() as tmp_results:
            from .config import LGCoTrainConfig

            config = LGCoTrainConfig(
                event=event,
                budget=budget,
                seed_set=seed_set,
                lr=lr,
                batch_size=batch_size,
                cotrain_epochs=cotrain_epochs,
                finetune_patience=finetune_patience,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                device=device,
                data_root=data_root,
                results_root=tmp_results,
                pseudo_label_source=pseudo_label_source,
            )

            if _trainer_cls is None:
                from .trainer import LGCoTrainer
                trainer = LGCoTrainer(config)
            else:
                trainer = _trainer_cls(config)

            result = trainer.run()

        return result["dev_macro_f1"]

    return objective


def run_single_study(
    event: str,
    budget: int,
    seed_set: int,
    n_trials: int,
    device: Optional[str] = None,
    storage_dir: str = "/workspace/results/optuna/per_experiment",
    data_root: str = "/workspace/data",
    pseudo_label_source: str = "gpt-4o",
    on_trial_done: Optional[Callable] = None,
    _trainer_cls=None,
) -> dict:
    """Run one Optuna study for a single (event, budget, seed_set).

    Creates an in-memory Optuna study, runs *n_trials* trials, saves
    ``best_params.json`` with best parameters, best value, and all trial
    results.  If ``best_params.json`` already exists, the study is skipped
    and the existing results are returned.

    Parameters
    ----------
    on_trial_done : callable, optional
        Called after each trial with ``(trial_number, n_trials, dev_f1)``.

    Returns
    -------
    dict with keys: event, budget, seed_set, status, best_params, best_value,
    n_trials, trials.
    """
    # Check for existing results (resume) — before importing optuna
    output_dir = Path(storage_dir) / event / f"{budget}_set{seed_set}"
    best_params_path = output_dir / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            return json.load(f)

    import optuna  # lazy — not needed at module level

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"{event}_b{budget}_s{seed_set}",
        direction="maximize",
    )

    # Wrap objective with callback
    base_objective = create_per_experiment_objective(
        event=event,
        budget=budget,
        seed_set=seed_set,
        device=device,
        data_root=data_root,
        pseudo_label_source=pseudo_label_source,
        _trainer_cls=_trainer_cls,
    )

    def objective_with_callback(trial):
        dev_f1 = base_objective(trial)
        if on_trial_done is not None:
            on_trial_done(trial.number, n_trials, dev_f1)
        return dev_f1

    study.optimize(objective_with_callback, n_trials=n_trials)

    # Build result
    trials_info = []
    for t in study.trials:
        trial_info = {
            "number": t.number,
            "state": t.state.name,
            "params": t.params,
        }
        if t.value is not None:
            trial_info["dev_macro_f1"] = round(t.value, 6)
        if t.datetime_start and t.datetime_complete:
            trial_info["duration_seconds"] = round(
                (t.datetime_complete - t.datetime_start).total_seconds(), 1
            )
        trials_info.append(trial_info)

    result = {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "status": "done",
        "best_params": study.best_params,
        "best_value": round(study.best_value, 6),
        "n_trials": len(study.trials),
        "trials": trials_info,
    }

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _run_study_worker(kwargs: dict) -> dict:
    """Worker function: run one Optuna study in a child process.

    Receives a plain dict (picklable), imports inside the function.
    Same pattern as ``parallel._run_single_experiment``.
    """
    event = kwargs["event"]
    budget = kwargs["budget"]
    seed_set = kwargs["seed_set"]

    try:
        return run_single_study(
            event=event,
            budget=budget,
            seed_set=seed_set,
            n_trials=kwargs["n_trials"],
            device=kwargs.get("device"),
            storage_dir=kwargs["storage_dir"],
            data_root=kwargs["data_root"],
            pseudo_label_source=kwargs.get("pseudo_label_source", "gpt-4o"),
        )
    except Exception as e:
        logging.getLogger("lg_cotrain").error(
            f"Optuna study {event} budget={budget} seed={seed_set} failed: {e}"
        )
        return {
            "event": event,
            "budget": budget,
            "seed_set": seed_set,
            "status": "failed",
            "best_params": None,
            "best_value": None,
            "n_trials": 0,
            "trials": [],
        }
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_all_studies(
    events: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    seed_sets: Optional[List[int]] = None,
    n_trials: int = 15,
    num_gpus: int = 1,
    storage_dir: str = "/workspace/results/optuna/per_experiment",
    data_root: str = "/workspace/data",
    pseudo_label_source: str = "gpt-4o",
    on_study_done: Optional[Callable] = None,
) -> List[dict]:
    """Run per-experiment Optuna studies for all combinations.

    Studies whose ``best_params.json`` already exists are skipped.
    Pending studies run in parallel across GPUs.

    Parameters
    ----------
    on_study_done : callable, optional
        Called after each study with ``(event, budget, seed_set, status)``.

    Returns
    -------
    List of result dicts in original (event, budget, seed_set) order.
    """
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    total = len(events) * len(budgets) * len(seed_sets)
    start_time = time.time()

    # Separate skipped from pending
    pre_results: Dict[Tuple[str, int, int], dict] = {}
    pending_configs: List[dict] = []
    pending_keys: List[Tuple[str, int, int]] = []
    skipped = 0

    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                best_params_path = (
                    Path(storage_dir) / event / f"{budget}_set{seed_set}" / "best_params.json"
                )
                if best_params_path.exists():
                    with open(best_params_path) as f:
                        result = json.load(f)
                    pre_results[(event, budget, seed_set)] = result
                    skipped += 1
                    print(
                        f"  {event} budget={budget} seed={seed_set}"
                        f" -- SKIPPED (exists)"
                    )
                    if on_study_done is not None:
                        on_study_done(event, budget, seed_set, "skipped")
                else:
                    pending_configs.append(dict(
                        event=event,
                        budget=budget,
                        seed_set=seed_set,
                        n_trials=n_trials,
                        storage_dir=storage_dir,
                        data_root=data_root,
                        pseudo_label_source=pseudo_label_source,
                    ))
                    pending_keys.append((event, budget, seed_set))

    print(
        f"\nOptuna per-experiment: {total} total, {skipped} skipped, "
        f"{len(pending_configs)} pending"
    )

    # Run pending studies
    if pending_configs:
        if num_gpus > 1:
            parallel_results = _run_studies_parallel(
                pending_configs, pending_keys, num_gpus, on_study_done,
            )
        else:
            parallel_results = _run_studies_sequential(
                pending_configs, pending_keys, on_study_done,
            )
    else:
        parallel_results = {}

    # Merge in original order
    all_results = []
    completed = failed = 0
    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                key = (event, budget, seed_set)
                if key in pre_results:
                    all_results.append(pre_results[key])
                elif key in parallel_results:
                    result = parallel_results[key]
                    all_results.append(result)
                    if result["status"] == "done":
                        completed += 1
                    else:
                        failed += 1

    # Write summary.json
    summary = {
        "total_studies": total,
        "completed": completed + skipped,
        "failed": failed,
        "n_trials_per_study": n_trials,
        "search_space": {
            "lr": "1e-5 to 1e-3 (log-uniform)",
            "batch_size": [8, 16, 32, 64],
            "cotrain_epochs": "5 to 20",
            "finetune_patience": "4 to 10",
            "weight_decay": "0.0 to 0.1",
            "warmup_ratio": "0.0 to 0.3",
        },
        "studies": [
            {
                "event": r["event"],
                "budget": r["budget"],
                "seed_set": r["seed_set"],
                "status": r["status"],
                "best_params": r.get("best_params"),
                "best_value": r.get("best_value"),
            }
            for r in all_results
        ],
    }
    summary_path = Path(storage_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time
    print(
        f"\nAll studies complete: {completed} ran, {skipped} skipped, "
        f"{failed} failed ({elapsed:.1f}s total)"
    )
    print(f"Summary saved to: {summary_path}")

    return all_results


def _run_studies_parallel(
    pending_configs: List[dict],
    pending_keys: List[Tuple[str, int, int]],
    num_gpus: int,
    on_study_done: Optional[Callable],
) -> Dict[Tuple[str, int, int], dict]:
    """Dispatch studies across GPUs using ProcessPoolExecutor."""
    # Assign GPUs round-robin
    for i, cfg in enumerate(pending_configs):
        cfg["device"] = f"cuda:{i % num_gpus}"

    print(
        f"Running {len(pending_configs)} Optuna studies in parallel "
        f"across {num_gpus} GPUs..."
    )

    ctx = mp.get_context("spawn")
    results_map: Dict[int, dict] = {}

    with ProcessPoolExecutor(
        max_workers=num_gpus, mp_context=ctx
    ) as executor:
        future_to_idx = {}
        for idx, cfg in enumerate(pending_configs):
            future = executor.submit(_run_study_worker, cfg)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            key = pending_keys[idx]
            try:
                result = future.result()
            except Exception as e:
                cfg = pending_configs[idx]
                result = {
                    "event": cfg["event"],
                    "budget": cfg["budget"],
                    "seed_set": cfg["seed_set"],
                    "status": "failed",
                    "best_params": None,
                    "best_value": None,
                    "n_trials": 0,
                    "trials": [],
                }
                logger.error(f"Process-level failure for study {idx}: {e}")

            results_map[idx] = result

            status = result["status"]
            best_val = result.get("best_value")
            val_str = f" (best_dev_f1={best_val:.4f})" if best_val else ""
            print(
                f"  {key[0]} budget={key[1]} seed={key[2]}"
                f" -- {status}{val_str}"
            )

            if on_study_done is not None:
                on_study_done(key[0], key[1], key[2], status)

    # Build lookup by key
    return {pending_keys[i]: results_map[i] for i in range(len(pending_configs))}


def _run_studies_sequential(
    pending_configs: List[dict],
    pending_keys: List[Tuple[str, int, int]],
    on_study_done: Optional[Callable],
) -> Dict[Tuple[str, int, int], dict]:
    """Run studies sequentially (num_gpus=1)."""
    results = {}
    for idx, (cfg, key) in enumerate(zip(pending_configs, pending_keys)):
        print(
            f"  [{idx + 1}/{len(pending_configs)}] "
            f"{key[0]} budget={key[1]} seed={key[2]} -- starting..."
        )
        result = _run_study_worker(cfg)
        results[key] = result

        status = result["status"]
        best_val = result.get("best_value")
        val_str = f" (best_dev_f1={best_val:.4f})" if best_val else ""
        print(
            f"  [{idx + 1}/{len(pending_configs)}] "
            f"{key[0]} budget={key[1]} seed={key[2]}"
            f" -- {status}{val_str}"
        )

        if on_study_done is not None:
            on_study_done(key[0], key[1], key[2], status)

    return results


def load_best_params(
    storage_dir: str = "/workspace/results/optuna/per_experiment",
    events: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    seed_sets: Optional[List[int]] = None,
) -> Dict[Tuple[str, int, int], dict]:
    """Load all best_params.json files into a dict.

    Returns
    -------
    Dict keyed by ``(event, budget, seed_set)`` with values being the
    full result dict (including ``best_params``, ``best_value``, etc.).
    """
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    results = {}
    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                path = (
                    Path(storage_dir) / event / f"{budget}_set{seed_set}" / "best_params.json"
                )
                if path.exists():
                    with open(path) as f:
                        results[(event, budget, seed_set)] = json.load(f)

    return results


def main():
    """CLI entry point: ``python -m lg_cotrain.optuna_per_experiment``."""
    parser = argparse.ArgumentParser(
        description="Per-experiment Optuna hyperparameter tuner for LG-CoTrain. "
        "Runs 120 separate studies (one per event/budget/seed combination) to "
        "find experiment-specific optimal hyperparameters.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Number of Optuna trials per study (default: 15)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for parallel study execution (default: 1)",
    )
    parser.add_argument(
        "--events", type=str, nargs="*", default=None,
        help="Events to tune (default: all 10)",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="*", default=None,
        help="Budgets to tune (default: all [5, 10, 25, 50])",
    )
    parser.add_argument(
        "--seed-sets", type=int, nargs="*", default=None,
        help="Seed sets to tune (default: all [1, 2, 3])",
    )
    parser.add_argument(
        "--data-root", type=str, default="/workspace/data",
    )
    parser.add_argument(
        "--storage-dir", type=str,
        default="/workspace/results/optuna/per_experiment",
        help="Directory for storing study results (default: results/optuna/per_experiment)",
    )
    parser.add_argument(
        "--pseudo-label-source", type=str, default="gpt-4o",
        help="Pseudo-label directory name (default: gpt-4o)",
    )

    args = parser.parse_args()

    run_all_studies(
        events=args.events,
        budgets=args.budgets,
        seed_sets=args.seed_sets,
        n_trials=args.n_trials,
        num_gpus=args.num_gpus,
        storage_dir=args.storage_dir,
        data_root=args.data_root,
        pseudo_label_source=args.pseudo_label_source,
    )


if __name__ == "__main__":
    main()
