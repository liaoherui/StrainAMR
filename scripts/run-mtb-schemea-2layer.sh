#!/usr/bin/env bash
set -eo pipefail

BASE=/data2/hmy/StrainAMR-optimize
GPU=${GPU:-1}
PY=/home/lhr/miniconda3/envs/strainamr/bin/python
STRAINAMR=/home/lhr/miniconda3/envs/strainamr/bin/strainamr
SCHEME_SCRIPT=$BASE/scripts/mtb_schemea_features_v2.py
FILTER_SCRIPT=$BASE/scripts/filter_feature_fold.py
REFERENCE=$BASE/references/Mtb/H37Rv_NC_000962.3.fna

FEATURE_ROOT=$BASE/features-mtb-schemeA
CLEAN_ROOT=$BASE/features-3fold-clean
SCHEME_MODEL_ROOT=$BASE/models-mtb-schemeA
BASELINE_EXTRA_ROOT=$BASE/models-baseline-extra-3fold
BASELINE_MODEL_ROOT=$BASE/models-3fold
LOG_ROOT=$BASE/runs-mtb-schemea-2layer/logs

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

mkdir -p "$LOG_ROOT"

log() {
  printf '[queue %s] %s\n' "$(date '+%F %T')" "$*"
}

nonempty() {
  [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

require_file() {
  if [ ! -s "$1" ]; then
    log "ERROR missing required file: $1"
    exit 1
  fi
}

feature_dir() {
  local drug=$1 fold=$2
  printf '%s/%s/Fold%s-run1' "$FEATURE_ROOT" "$drug" "$fold"
}

feature_drug() {
  case "$1" in
    Mtb_ofloxacin_3fold) printf 'ofloxacin' ;;
    Mtb_ciprofloxacin_3fold) printf 'ciprofloxacin' ;;
    *) log "ERROR unsupported drug dataset: $1"; exit 1 ;;
  esac
}

clean_dir() {
  local drug=$1 fold=$2
  printf '%s/%s/Fold%s' "$CLEAN_ROOT" "$drug" "$fold"
}

scheme_model_dir() {
  local drug=$1 fold=$2
  printf '%s/%s/Fold%s/model-gpu%s' "$SCHEME_MODEL_ROOT" "$drug" "$fold" "$GPU"
}

build_features() {
  local drug=$1 fold=$2 exclude=$3
  local fold_dir=$BASE/features-3fold/$drug/Fold$fold
  local out
  local log_file=$LOG_ROOT/${drug}-Fold${fold}.feature.log
  out=$(feature_dir "$drug" "$fold")

  if [ -f "$out/.feature.complete" ]; then
    log "feature skip $drug Fold$fold"
    return
  fi
  if nonempty "$out"; then
    log "ERROR feature output is nonempty without completion marker: $out"
    exit 1
  fi

  local args=(
    "$SCHEME_SCRIPT"
    --fold-dir "$fold_dir"
    --out-dir "$out"
    --reference "$REFERENCE"
    --reference-name NC_000962.3
    --drug "$(feature_drug "$drug")"
    --threads 1 --jobs 8
  )
  if [ -n "$exclude" ]; then
    args+=(--exclude-sample "$exclude")
  fi

  log "feature start $drug Fold$fold -> $out"
  "$PY" "${args[@]}" >"$log_file" 2>&1
  require_file "$out/mtb_feature_qc.tsv"
  require_file "$out/val_features/strains_test_sentence_fs.txt"
  touch "$out/.feature.complete"
  log "feature done $drug Fold$fold"
}

build_clean_baseline() {
  local drug=$1 fold=$2 exclude=$3
  local src=$BASE/features-3fold/$drug/Fold$fold
  local out
  local log_file=$LOG_ROOT/${drug}-Fold${fold}.filter-baseline.log
  out=$(clean_dir "$drug" "$fold")

  if [ -f "$out/.filter.complete" ]; then
    log "filtered baseline skip $drug Fold$fold"
    return
  fi
  if nonempty "$out"; then
    log "ERROR clean feature output is nonempty without completion marker: $out"
    exit 1
  fi

  log "filtered baseline start $drug Fold$fold -> $out"
  "$PY" "$FILTER_SCRIPT" \
    --fold-dir "$src" \
    --out-dir "$out" \
    --exclude-sample "$exclude" \
    >"$log_file" 2>&1
  require_file "$out/filter_manifest.json"
  touch "$out/.filter.complete"
  log "filtered baseline done $drug Fold$fold"
}

train_model() {
  local features=$1 val_features=$2 model=$3 tag=$4
  local log_file=$LOG_ROOT/${tag}.train.log

  if [ -s "$model/models/best_model_f1_score.pt" ]; then
    log "train skip $tag"
    return
  fi
  if nonempty "$model"; then
    log "ERROR model output is nonempty without checkpoint: $model"
    exit 1
  fi

  log "train start $tag on GPU$GPU"
  "$STRAINAMR" train \
    -i "$features" \
    --val-dir "$val_features" \
    -o "$model" \
    -s 1 --patience 20 --epochs 100 \
    >"$log_file" 2>&1
  require_file "$model/models/best_model_f1_score.pt"
  require_file "$model/logs/output_sample_prob_val_best_f1.txt"
  touch "$model/.train.complete"
  log "train done $tag"
}

predict_model() {
  local features=$1 model=$2 pred=$3 tag=$4
  local log_file=$LOG_ROOT/${tag}.predict.log

  if [ -s "$pred/logs/output_sample_prob_predict.txt" ]; then
    log "predict skip $tag"
    return
  fi
  if nonempty "$pred"; then
    log "ERROR prediction output is nonempty without result: $pred"
    exit 1
  fi

  log "predict start $tag"
  "$STRAINAMR" predict \
    -i "$features/val_features" \
    -m "$model" \
    -o "$pred" \
    --db "$features/db" \
    >"$log_file" 2>&1
  require_file "$pred/logs/output_sample_prob_predict.txt"
  touch "$pred/.predict.complete"
  log "predict done $tag"
}

ensure_baseline() {
  local drug=$1 fold=$2 exclude=$3
  local existing
  existing=$(ls "$BASELINE_MODEL_ROOT/$drug/Fold$fold"/model-gpu*/models/best_model_f1_score.pt 2>/dev/null | head -n 1 || true)
  if [ -n "$existing" ]; then
    log "baseline exists $drug Fold$fold: $existing"
    return
  fi

  local features val_features model tag
  if [ -n "$exclude" ]; then
    build_clean_baseline "$drug" "$fold" "$exclude"
    local clean
    clean=$(clean_dir "$drug" "$fold")
    features=$clean/db
    val_features=$clean/val_features
  else
    features=$BASE/features-3fold/$drug/Fold$fold/db
    val_features=$BASE/features-3fold/$drug/Fold$fold/val_features
  fi
  model=$BASELINE_EXTRA_ROOT/$drug/Fold$fold/model-gpu$GPU
  tag=${drug}-Fold${fold}.baseline-extra
  train_model "$features" "$val_features" "$model" "$tag"
}

run_fold() {
  local drug=$1 fold=$2 exclude=$3
  local feat model pred tag
  feat=$(feature_dir "$drug" "$fold")
  model=$(scheme_model_dir "$drug" "$fold")
  pred=$SCHEME_MODEL_ROOT/$drug/Fold$fold/prediction-gpu$GPU
  tag=${drug}-Fold${fold}.schemeA

  build_features "$drug" "$fold" "$exclude"
  ensure_baseline "$drug" "$fold" "$exclude"
  train_model "$feat/db" "$feat/val_features" "$model" "$tag"
  predict_model "$feat" "$model" "$pred" "$tag"
}

log "two-layer queue start on GPU$GPU"
log "layer 1: ofloxacin Fold2/3"
run_fold Mtb_ofloxacin_3fold 2 mtb_1324261
run_fold Mtb_ofloxacin_3fold 3 mtb_1324261

log "layer 2: ciprofloxacin Fold1/2/3"
run_fold Mtb_ciprofloxacin_3fold 1 ""
run_fold Mtb_ciprofloxacin_3fold 2 ""
run_fold Mtb_ciprofloxacin_3fold 3 ""

touch "$BASE/runs-mtb-schemea-2layer/.queue.complete"
log "two-layer queue complete"
