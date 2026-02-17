"""Evaluation metrics and ensemble prediction."""

from collections import Counter
from typing import Dict, List


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute error rate, macro-F1, and per-class F1.

    Args:
        y_true: Ground truth label ids (list or array).
        y_pred: Predicted label ids (list or array).

    Returns:
        Dict with error_rate (%), macro_f1, and per_class_f1 (list).
    """
    try:
        import numpy as np
        from sklearn.metrics import f1_score
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error_rate = 100.0 * (1.0 - np.mean(y_true == y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    except ImportError:
        # Pure-Python fallback
        n = len(y_true)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        error_rate = 100.0 * (1.0 - correct / n) if n > 0 else 0.0
        macro_f1, per_class_f1 = _compute_f1_pure(list(y_true), list(y_pred))

    return {
        "error_rate": error_rate,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


def _compute_f1_pure(y_true: list, y_pred: list):
    """Pure-Python macro-F1 and per-class F1 computation."""
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_f1, f1s


def ensemble_predict(model1, model2, loader, device):
    """Ensemble prediction: average softmax of two models, then argmax.

    Requires torch. Returns (predictions, true_labels) as numpy arrays.
    """
    import numpy as np
    import torch

    model1.eval()
    model2.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            probs1 = model1.predict_proba(input_ids, attention_mask)
            probs2 = model2.predict_proba(input_ids, attention_mask)

            avg_probs = (probs1 + probs2) / 2.0
            preds = avg_probs.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_pseudo_labels(class_label, predicted_label) -> float:
    """Compute accuracy of pseudo-labels against ground truth."""
    n = len(class_label)
    correct = sum(1 for a, b in zip(class_label, predicted_label) if a == b)
    return (correct / n) * 100.0 if n > 0 else 0.0
