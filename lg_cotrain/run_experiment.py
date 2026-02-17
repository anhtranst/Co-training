"""CLI entry point for running LG-CoTrain experiments."""

import argparse

from .config import LGCoTrainConfig
from .trainer import LGCoTrainer


def main():
    parser = argparse.ArgumentParser(description="LG-CoTrain experiment runner")
    parser.add_argument("--event", type=str, default="canada_wildfires_2016")
    parser.add_argument("--budget", type=int, default=5, choices=[5, 10, 25, 50])
    parser.add_argument("--seed-set", type=int, default=1, choices=[1, 2, 3])
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

    config = LGCoTrainConfig(
        event=args.event,
        budget=args.budget,
        seed_set=args.seed_set,
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

    trainer = LGCoTrainer(config)
    results = trainer.run()

    print(f"\nFinal Results:")
    print(f"  Test Error Rate: {results['test_error_rate']:.2f}%")
    print(f"  Test Macro-F1:   {results['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
