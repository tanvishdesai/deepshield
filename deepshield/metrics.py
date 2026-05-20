from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn import metrics


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[idx] + fpr[idx]) / 2.0)


def classification_metrics(y_true: Iterable[int], y_score: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_score = np.asarray(list(y_score), dtype=np.float64)
    if y_true.size == 0:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "eer": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "count": 0.0,
        }
    y_pred = (y_score >= 0.5).astype(np.int64)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        ap = float("nan")
        eer = float("nan")
    else:
        auc = float(metrics.roc_auc_score(y_true, y_score))
        ap = float(metrics.average_precision_score(y_true, y_score))
        eer = compute_eer(y_true, y_score)

    return {
        "auc": auc,
        "ap": ap,
        "eer": eer,
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "count": float(len(y_true)),
    }


def confidence_interval95(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "std": float("nan")}
    mean = float(np.mean(values))
    if values.size == 1:
        return {"mean": mean, "ci95": 0.0, "std": 0.0}
    std = float(np.std(values, ddof=1))
    ci95 = 1.96 * std / math.sqrt(values.size)
    return {"mean": mean, "ci95": float(ci95), "std": std}
