"""Batch runner: execute all 12 (budget x seed_set) experiments for one event."""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from .config import LGCoTrainConfig

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]

logger = logging.getLogger("lg_cotrain")


def run_all_experiments(
    event,
    *,
    model_name="bert-base-uncased",
    weight_gen_epochs=7,
    cotrain_epochs=10,
    finetune_max_epochs=100,
    finetune_patience=5,
    batch_size=32,
    lr=2e-5,
    max_seq_length=128,
    data_root="/workspace/data",
    results_root="/workspace/results",
    _trainer_cls=None,
):
    """Run all budget x seed_set combinations for *event*.

    Returns a list of 12 result dicts (or ``None`` for failed experiments).
    Experiments whose ``metrics.json`` already exists are loaded and skipped.
    """
    if _trainer_cls is None:
        from .trainer import LGCoTrainer  # lazy — avoids torch import at module level

        _trainer_cls = LGCoTrainer

    all_results = []
    total = len(BUDGETS) * len(SEED_SETS)
    completed = skipped = failed = 0
    start_time = time.time()

    for budget in BUDGETS:
        for seed_set in SEED_SETS:
            idx = completed + skipped + failed + 1
            metrics_path = (
                Path(results_root) / event / f"{budget}_set{seed_set}" / "metrics.json"
            )

            # Resume: reuse existing results
            if metrics_path.exists():
                with open(metrics_path) as f:
                    result = json.load(f)
                all_results.append(result)
                skipped += 1
                print(
                    f"[{idx}/{total}] budget={budget}, seed={seed_set}"
                    f" -- SKIPPED (exists)"
                )
                continue

            print(f"[{idx}/{total}] budget={budget}, seed={seed_set} -- starting...")
            config = LGCoTrainConfig(
                event=event,
                budget=budget,
                seed_set=seed_set,
                model_name=model_name,
                weight_gen_epochs=weight_gen_epochs,
                cotrain_epochs=cotrain_epochs,
                finetune_max_epochs=finetune_max_epochs,
                finetune_patience=finetune_patience,
                batch_size=batch_size,
                lr=lr,
                max_seq_length=max_seq_length,
                data_root=data_root,
                results_root=results_root,
            )

            try:
                trainer = _trainer_cls(config)
                result = trainer.run()
                all_results.append(result)
                completed += 1
                print(
                    f"[{idx}/{total}] budget={budget}, seed={seed_set}"
                    f" -- done (macro_f1={result['test_macro_f1']:.4f})"
                )
            except Exception as e:
                logger.error(
                    f"Experiment budget={budget}, seed={seed_set} failed: {e}"
                )
                all_results.append(None)
                failed += 1

            # Free GPU memory between experiments
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    elapsed = time.time() - start_time
    print(
        f"\nBatch complete: {completed} ran, {skipped} skipped, {failed} failed"
        f" ({elapsed:.1f}s total)"
    )
    return all_results


def format_summary_table(all_results, event):
    """Return a formatted summary table grouped by budget."""
    # Build lookup: (budget, seed_set) -> result dict
    lookup = {}
    for r in all_results:
        if r is not None:
            lookup[(r["budget"], r["seed_set"])] = r

    lines = []
    lines.append(f"=== Results for {event} ===")
    lines.append("")

    # Header
    seed_hdrs = "".join(f"  Seed {s:<13}" for s in SEED_SETS)
    lines.append(f"{'Budget':>6}  {seed_hdrs}  {'Mean':>8}  {'Std':>8}")

    sub_cells = "".join(f"  {'ErrR%':>6} {'MacF1':>6}" for _ in SEED_SETS)
    lines.append(f"{'':>6}  {sub_cells}  {'ErrR%':>8}  {'MacF1':>8}")
    lines.append("-" * len(lines[-1]))

    for budget in BUDGETS:
        err_rates = []
        macro_f1s = []
        cells = ""
        for seed_set in SEED_SETS:
            r = lookup.get((budget, seed_set))
            if r is not None:
                cells += f"  {r['test_error_rate']:>6.2f} {r['test_macro_f1']:>6.4f}"
                err_rates.append(r["test_error_rate"])
                macro_f1s.append(r["test_macro_f1"])
            else:
                cells += f"  {'N/A':>6} {'N/A':>6}"

        # Mean ± std
        if len(err_rates) >= 2:
            e_mean = statistics.mean(err_rates)
            e_std = statistics.stdev(err_rates)
            f_mean = statistics.mean(macro_f1s)
            f_std = statistics.stdev(macro_f1s)
            agg = f"  {e_mean:>5.2f}+/-{e_std:<5.2f}  {f_mean:.4f}+/-{f_std:.4f}"
        elif len(err_rates) == 1:
            agg = f"  {err_rates[0]:>8.2f}  {macro_f1s[0]:>8.4f}"
        else:
            agg = f"  {'N/A':>8}  {'N/A':>8}"

        lines.append(f"{budget:>6}  {cells}{agg}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run all 12 (budget x seed_set) experiments for one event"
    )
    parser.add_argument(
        "--event", type=str, required=True,
        help="Disaster event name, e.g. canada_wildfires_2016",
    )
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--weight-gen-epochs", type=int, default=7)
    parser.add_argument("--cotrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-max-epochs", type=int, default=100)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="/workspace/data")
    parser.add_argument("--results-root", type=str, default="/workspace/results")

    args = parser.parse_args()

    all_results = run_all_experiments(
        args.event,
        model_name=args.model_name,
        weight_gen_epochs=args.weight_gen_epochs,
        cotrain_epochs=args.cotrain_epochs,
        finetune_max_epochs=args.finetune_max_epochs,
        finetune_patience=args.finetune_patience,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        data_root=args.data_root,
        results_root=args.results_root,
    )

    print()
    print(format_summary_table(all_results, args.event))


if __name__ == "__main__":
    main()
