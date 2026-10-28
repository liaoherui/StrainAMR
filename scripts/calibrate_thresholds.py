#!/usr/bin/env python3
"""Turn an eval-thresholds run directory into a per-drug threshold config.

For every (source, dataset) group it reports
  * oracle_threshold  : best F1 on that fold (optimistic, not deployable)
  * loo_threshold     : median of the *other* folds' oracle thresholds
  * global_threshold  : median of all folds' oracle thresholds (what you would
                        actually ship for that drug)
and the F1 each of those thresholds achieves, fold by fold.

Reads only the prediction files referenced by the run JSONs; writes a new
JSON + markdown file, never touches model directories.
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_threshold_metrics as ev  # noqa: E402


def r6(v):
    return None if v is None else round(v, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="<eval dir>/runs produced by eval_threshold_metrics.py")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--md", default=None, help="optional markdown path")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.runs, "*.json")))
    if not files:
        print("no run json under", args.runs, file=sys.stderr)
        return 2

    groups = {}
    for path in files:
        with open(path) as fh:
            rec = json.load(fh)
        groups.setdefault((rec["source"], rec["dataset"]), []).append(rec)

    payload = {
        "note": ("global_threshold = median of the per-fold oracle thresholds; "
                 "loo_threshold = median of the other folds' oracle thresholds. "
                 "With very few positives per fold these medians are unstable."),
        "datasets": {},
    }
    md_rows = []
    for (source, dataset), recs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        recs = sorted(recs, key=lambda r: r["fold"])
        thr = {str(r["fold"]): r["oracle_threshold"] for r in recs}
        global_thr = statistics.median(thr.values())
        folds = {}
        f1_global, f1_05, f1_oracle = [], [], []
        for rec in recs:
            rows = ev.read_probs(rec["file"])
            y = [x[1] for x in rows]
            s = [x[2] for x in rows]
            others = [t for f, t in thr.items() if str(f) != str(rec["fold"])]
            loo = statistics.median(others) if others else None
            g = ev.confusion(y, s, global_thr)
            loo_m = ev.confusion(y, s, loo) if loo is not None else None
            folds[str(rec["fold"])] = {
                "n": len(y), "n_pos": sum(y),
                "oracle_threshold": r6(rec["oracle_threshold"]),
                "f1_at_oracle": r6(rec["oracle"]["f1"]),
                "f1_at_0.5": r6(rec["at_0.5"]["f1"]),
                "loo_threshold": r6(loo),
                "f1_at_loo": r6(loo_m["f1"]) if loo_m else None,
                "f1_at_global": r6(g["f1"]),
                "sens_at_global": r6(g["sensitivity"]),
                "spec_at_global": r6(g["specificity"]),
            }
            f1_global.append(g["f1"])
            f1_05.append(rec["at_0.5"]["f1"])
            f1_oracle.append(rec["oracle"]["f1"])
        payload["datasets"][dataset] = {
            "source": source,
            "n_folds": len(recs),
            "global_threshold": r6(global_thr),
            "oracle_thresholds": {k: r6(v) for k, v in sorted(thr.items())},
            "oracle_threshold_range": [r6(min(thr.values())), r6(max(thr.values()))],
            "mean_f1_at_0.5": r6(statistics.mean(f1_05)),
            "mean_f1_at_oracle": r6(statistics.mean(f1_oracle)),
            "mean_f1_at_global": r6(statistics.mean(f1_global)),
            "folds": folds,
        }
        md_rows.append((dataset, source, len(recs), statistics.mean(f1_05), statistics.mean(f1_oracle),
                        global_thr, min(thr.values()), max(thr.values()), statistics.mean(f1_global)))

    with open(args.out, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("wrote", args.out)

    if args.md:
        md = ["# Calibrated thresholds", "",
              "`global_thr` = median of the per-fold oracle thresholds (a shippable per-drug threshold).",
              "`thr_oracle` per fold is optimistic; the spread across folds is the honest uncertainty.", "",
              "| dataset | source | folds | F1@0.5 | F1@oracle | global_thr | thr range | F1@global |",
              "|---|---|---|---|---|---|---|---|"]
        for row in md_rows:
            md.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f}–{:.3f} | {:.3f} |".format(*row))
        with open(args.md, "w") as fh:
            fh.write("\n".join(md) + "\n")
        print("wrote", args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
