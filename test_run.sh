#!/usr/bin/env bash
# End-to-end demo on the small S. aureus / levofloxacin set shipped in Test_genomes.zip
# (61 training genomes, 5 validation genomes).
#
#   bash test_run.sh
#
# Steps: unpack -> build-train -> build-test -> train -> predict.
# Everything is written to Test_run/ and Model_test/, never into the repository.
# Runtime is dominated by gene calling and RGI annotation; expect tens of minutes on a laptop.
#
# Environment variables: THREADS (default 8).
#
# The original version of this script called the four Python entry points directly and passed
# Model_test/Sau_levofloxacin/checkpoint_best.pt to predict. Training now writes
# models/best_model_f1_score.pt and records it in selected_model.txt, so the model *directory*
# is what predict expects. build-test also needs an explicit --db, and build-train needs
# --amr-organism for the known-mechanism layer.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

THREADS=${THREADS:-8}
# Multi-GPU training goes through NCCL, which fails on shared or mixed-driver hosts.
# Restrict the demo to one card unless the caller says otherwise.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRUG=levofloxacin
ORGANISM=Staphylococcus_aureus
DATA=Test_genomes/Sau
DB=Test_run/db
VAL=Test_run/val_features
MODEL=Model_test/Sau_levofloxacin
PRED=Test_run/predictions

step() { printf '\n=== %s ===\n' "$*"; }

if [ -f strainamr.py ]; then
  run() { python -u strainamr.py "$@"; }
elif command -v strainamr >/dev/null 2>&1; then
  run() { strainamr "$@"; }
else
  echo "strainamr not found: run 'pip install -e .' or execute this script from the repository root" >&2
  exit 1
fi

if [ ! -d "$DATA" ]; then
  step "unzip Test_genomes.zip"
  unzip -q Test_genomes.zip
fi

if [ ! -f localDB/card.json ]; then
  step "unzip localDB.zip (CARD database used by RGI --local)"
  unzip -q localDB.zip
fi

if [ -f "$DB/node_token_match.txt" ]; then
  step "build-train: already done ($DB)"
else
step "build-train ($(find "$DATA/Ref_train" -name '*.fna' | wc -l) genomes, $(wc -l < "$DATA/sau_label_train.txt") labelled)"
run build-train \
  -i "$DATA/Ref_train" \
  -l "$DATA/sau_label_train.txt" \
  -d "$DRUG" \
  -o "$DB" \
  -t "$THREADS" \
  --amr-organism "$ORGANISM"
fi

if [ -f "$VAL/strains_test_sentence_fs.txt" ]; then
  step "build-test: already done ($VAL)"
else
step "build-test ($(find "$DATA/Ref_val" -name '*.fna' | wc -l) genomes)"
run build-test \
  -i "$DATA/Ref_val" \
  -l "$DATA/sau_label_val.txt" \
  -d "$DRUG" \
  -o "$VAL" \
  --db "$DB" \
  -t "$THREADS"
fi

if [ -f "$MODEL/models/best_model_f1_score.pt" ]; then
  step "train: already done ($MODEL)"
else
step "train"
run train \
  -i "$DB" \
  --val-dir "$VAL" \
  -o "$MODEL" \
  -s 1 --patience 20 --epochs 100
fi

step "predict"
run predict \
  -i "$VAL" \
  -m "$MODEL" \
  -o "$PRED" \
  --db "$DB" \
  --skip-analysis

step "done"
echo "model:       $MODEL/models/best_model_f1_score.pt"
echo "predictions: $PRED/predictions.tsv"
echo
echo "Metrics for these 5 validation genomes are printed by the predict step."
echo "Drop --skip-analysis to also get attribution and interaction output."
