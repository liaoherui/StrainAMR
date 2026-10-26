#!/usr/bin/env python3
"""Compare two StrainAMR prediction files on exactly the same samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys


def load_predictions(path: str, excluded: set[str]) -> dict[str, dict[str, str]]:
    rows = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample_key = next(
                (key for key in row if key.lower() == "sample_id"), None
            )
            if sample_key is None:
                raise ValueError(f"no Sample_ID column in {path}")
            sample_id = row[sample_key]
            if sample_id not in excluded:
                rows[sample_id] = row
    return rows


def calculate(rows: dict[str, dict[str, str]]) -> dict[str, float | int]:
    values = list(rows.values())
    tp = sum(row["Label"] == "1" and row["Pred"] == "1" for row in values)
    tn = sum(row["Label"] == "0" and row["Pred"] == "0" for row in values)
    fp = sum(row["Label"] == "0" and row["Pred"] == "1" for row in values)
    fn = sum(row["Label"] == "1" and row["Pred"] == "0" for row in values)
    accuracy = (tp + tn) / len(values)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0

    positive = [float(row["Prob"]) for row in values if row["Label"] == "1"]
    negative = [float(row["Prob"]) for row in values if row["Label"] == "0"]
    roc_auc = sum(
        (pos > neg) + 0.5 * (pos == neg)
        for pos in positive for neg in negative
    ) / (len(positive) * len(negative))
    ordered = sorted(values, key=lambda row: float(row["Prob"]), reverse=True)
    hits = 0
    average_precision = 0.0
    for rank, row in enumerate(ordered, 1):
        if row["Label"] == "1":
            hits += 1
            average_precision += hits / rank / len(positive)

    return {
        "n": len(values), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "balanced_accuracy": balanced_accuracy,
        "mcc": mcc, "roc_auc": roc_auc, "pr_auc": average_precision,
        "predicted_resistant_fraction": (tp + fp) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    excluded = set(args.exclude)
    baseline = load_predictions(args.baseline, excluded)
    scheme = load_predictions(args.scheme, excluded)
    if set(baseline) != set(scheme):
        missing_baseline = sorted(set(scheme) - set(baseline))
        missing_scheme = sorted(set(baseline) - set(scheme))
        raise ValueError(
            f"sample sets differ: missing baseline={missing_baseline[:5]}, "
            f"missing scheme={missing_scheme[:5]}"
        )

    baseline_metrics = calculate(baseline)
    scheme_metrics = calculate(scheme)
    flips = []
    for sample_id in sorted(baseline):
        left = baseline[sample_id]
        right = scheme[sample_id]
        if left["Pred"] != right["Pred"]:
            flips.append({
                "sample_id": sample_id,
                "label": left["Label"],
                "baseline_pred": left["Pred"],
                "baseline_probability": float(left["Prob"]),
                "scheme_pred": right["Pred"],
                "scheme_probability": float(right["Prob"]),
            })
    result = {
        "drug": args.drug,
        "fold": args.fold,
        "excluded": sorted(excluded),
        "baseline_file": args.baseline,
        "scheme_file": args.scheme,
        "baseline": baseline_metrics,
        "scheme": scheme_metrics,
        "delta_scheme_minus_baseline": {
            key: scheme_metrics[key] - baseline_metrics[key]
            for key in (
                "accuracy", "sensitivity", "specificity", "precision", "f1",
                "balanced_accuracy", "mcc", "roc_auc", "pr_auc",
                "predicted_resistant_fraction",
            )
        },
        "prediction_flips": flips,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
