from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict[str, object]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")

    labels = sorted(set(true_labels) | set(predicted_labels))
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    macro_f1 = float(sum(f1) / len(f1)) if len(f1) else 0.0

    per_label = {}
    for index, label in enumerate(labels):
        per_label[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "labels": labels,
        "per_label": per_label,
        "confusion_matrix": matrix.tolist(),
    }
