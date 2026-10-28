#!/usr/bin/env bash
# Train + predict the folds whose features are already complete, while the main
# queue is still busy building the amoxicillin features.
#
# Runs in parallel with run-remaining-folds-gpu.sh. Safe against duplication:
#   - skips a fold when its model or prediction already exists
#   - skips a fold when the main queue is currently training/predicting it
# Output dir names keep the historical "gpu1" suffix so the main queue's
# "model exists" check still sees them; the physical device is logged.
set -uo pipefail

workroot=/data2/hmy/StrainAMR-optimize
featureroot="$workroot/features-3fold"
modelroot="$workroot/models-3fold"
stamp=$(date '+%Y%m%d-%H%M')
rundir="$workroot/runs-train-ready-$stamp"
logdir="$rundir/logs"
gpu_pref="${GPU_PREF:-0 1}"
min_free_mb="${MIN_FREE_MB:-12000}"
poll_s="${POLL_S:-120}"

mkdir -p "$logdir"
log() { printf '[ready %s] %s\n' "$(date '+%F %T')" "$*"; }

if ! command -v strainamr >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source /home/lhr/miniconda3/etc/profile.d/conda.sh
    conda activate strainamr
    set -u
fi
command -v strainamr >/dev/null 2>&1 || { echo "strainamr not found" >&2; exit 1; }
cd "$workroot/code"

gpu_free_mb() {
    nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' 'NR==1 {printf "%d", $1 - $2}'
}

pick_gpu() {
    for g in $gpu_pref; do
        free=$(gpu_free_mb "$g"); [ -n "${free:-}" ] || continue
        if [ "$free" -ge "$min_free_mb" ]; then gpu=$g; return 0; fi
    done
    return 1
}

wait_gpu() {
    step=$1; waited=0
    while ! pick_gpu; do
        [ $(( waited % 600 )) -eq 0 ] && log "waiting for >=${min_free_mb}M free on [$gpu_pref] before $step (waited ${waited}s)"
        sleep "$poll_s"; waited=$(( waited + poll_s ))
    done
    log "using GPU$gpu (free $(gpu_free_mb "$gpu")M) for $step"
}

log "run dir: $rundir"
for target in \
    "Mtb_ethionamide_3fold Fold2" \
    "Mtb_ethionamide_3fold Fold3" \
    "Sau_clindamycin_3fold Fold3" \
    "Sau_levofloxacin_3fold Fold2" \
    "Sau_levofloxacin_3fold Fold3"
do
    set -- $target
    dataset=$1; fold=$2
    fdir="$featureroot/$dataset/$fold"
    base="$modelroot/$dataset/$fold"
    model="$base/model-gpu1"
    pred="$base/prediction-gpu1"
    mkdir -p "$base" "$logdir"

    if [ ! -f "$fdir/.build-train.complete" ] || [ ! -f "$fdir/.build-val.complete" ]; then
        log "SKIP $dataset/$fold: features incomplete"
        continue
    fi

    if pgrep -f "$dataset/$fold/model-gpu1" >/dev/null 2>&1 || pgrep -f "$dataset/$fold/prediction-gpu1" >/dev/null 2>&1; then
        log "SKIP $dataset/$fold: main queue is already working on it"
        continue
    fi

    if [ -f "$model/models/best_model_f1_score.pt" ]; then
        log "SKIP train $dataset/$fold: model exists"
    else
        wait_gpu "train $dataset/$fold"
        log "train start $dataset/$fold on GPU$gpu"
        CUDA_VISIBLE_DEVICES=$gpu strainamr train \
            -i "$fdir/db" \
            --val-dir "$fdir/val_features" \
            -o "$model" \
            -s 1 --patience 20 --epochs 100 \
            > "$logdir/${dataset}-${fold}-train.log" 2>&1
        if [ $? -ne 0 ] || [ ! -f "$model/models/best_model_f1_score.pt" ]; then
            log "train FAILED $dataset/$fold (see $logdir/${dataset}-${fold}-train.log)"
            continue
        fi
        log "train done $dataset/$fold"
    fi

    if [ -f "$pred/logs/output_sample_prob_predict.txt" ]; then
        log "SKIP predict $dataset/$fold: output exists"
        continue
    fi
    wait_gpu "predict $dataset/$fold"
    log "predict start $dataset/$fold on GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu strainamr predict \
        -i "$fdir/val_features" \
        -m "$model" \
        -o "$pred" \
        --db "$fdir/db" \
        > "$logdir/${dataset}-${fold}-predict.log" 2>&1
    if [ $? -ne 0 ] || [ ! -f "$pred/logs/output_sample_prob_predict.txt" ]; then
        log "predict FAILED $dataset/$fold (see $logdir/${dataset}-${fold}-predict.log)"
        continue
    fi
    log "predict done $dataset/$fold"
done

log "ALL REQUESTED FOLDS PROCESSED"
touch "$rundir/.ready.complete"
