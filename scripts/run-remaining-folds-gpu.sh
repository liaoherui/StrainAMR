#!/usr/bin/env bash
# Reuse the remaining Mtb folds, then train and predict everything that is missing.
#
# v2 (2026-10-21): adds a GPU gate. Before every train/predict step the queue
# waits until one of the candidate GPUs is nearly idle, i.e. other people's jobs
# have finished. Tune with GPU_CANDIDATES / GPU_MIN_FREE_MB / GPU_MAX_UTIL /
# GPU_WAIT_MAX_S / GPU_POLL_S. If nothing frees up before GPU_WAIT_MAX_S the
# step starts anyway on the least busy card, so the queue can never stall.
#
# Idempotent: completed features/models/predictions are skipped, nothing is overwritten.
set -uo pipefail

workroot=/data2/hmy/StrainAMR-optimize
featureroot="$workroot/features-3fold"
scriptdir="$workroot/scripts"
runstamp=$(date '+%Y%m%d-%H%M')
rundir="$workroot/runs-remaining-folds-$runstamp"
logdir="$rundir/logs"

gpu_candidates="${GPU_CANDIDATES:-1 0}"
gpu_min_free_mb="${GPU_MIN_FREE_MB:-36000}"
gpu_max_util="${GPU_MAX_UTIL:-60}"
gpu_wait_max_s="${GPU_WAIT_MAX_S:-21600}"
gpu_poll_s="${GPU_POLL_S:-120}"
gpu=1

mkdir -p "$logdir"
exec 9>"$rundir/.queue.lock"
flock -n 9 || { echo "another queue holds $rundir/.queue.lock" >&2; exit 1; }

log() { printf '[queue %s] %s\n' "$(date '+%F %T')" "$*"; }

if ! command -v strainamr >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source /home/lhr/miniconda3/etc/profile.d/conda.sh
    conda activate strainamr
    set -u
fi
command -v strainamr >/dev/null 2>&1 || { echo "strainamr not found" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found" >&2; exit 1; }
cd "$workroot/code"

# ------------------------------------------------------------------ GPU gate
gpu_free_mb() {
    nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' 'NR==1 {printf "%d", $1 - $2}'
}

gpu_util() {
    nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | head -n1 | tr -d ' '
}

pick_gpu() {
    best_gpu=""; best_free=-1
    for g in $gpu_candidates; do
        free=$(gpu_free_mb "$g"); [ -n "${free:-}" ] || continue
        util=$(gpu_util "$g"); [ -n "${util:-}" ] || util=100
        if [ "$free" -ge "$gpu_min_free_mb" ] && [ "$util" -le "$gpu_max_util" ]; then
            gpu=$g
            return 0
        fi
        if [ "$free" -gt "$best_free" ]; then best_free=$free; best_gpu=$g; fi
    done
    return 1
}

gpu_snapshot() {
    for g in $gpu_candidates; do
        printf 'GPU%s free=%sM util=%s%% ' "$g" "$(gpu_free_mb "$g")" "$(gpu_util "$g")"
    done
}

wait_for_gpu() {
    step=$1
    start=$(date +%s); n=0
    while :; do
        if pick_gpu; then
            [ "$n" -gt 0 ] && log "GPU free after $(( $(date +%s) - start ))s -> GPU$gpu before: $step  [$(gpu_snapshot)]"
            return 0
        fi
        now=$(date +%s); waited=$(( now - start ))
        if [ "$waited" -ge "$gpu_wait_max_s" ]; then
            gpu=$best_gpu
            log "GPU wait timed out after ${waited}s -> starting $step on least-busy GPU$gpu  [$(gpu_snapshot)]"
            return 0
        fi
        if [ $(( n % 10 )) -eq 0 ]; then
            log "waiting for a free GPU before $step (waited ${waited}s)  [$(gpu_snapshot)]"
        fi
        n=$(( n + 1 ))
        sleep "$gpu_poll_s"
    done
}

log "run dir: $rundir"
log "GPU gate: candidates=[$gpu_candidates] need free>=${gpu_min_free_mb}M util<=${gpu_max_util}% poll=${gpu_poll_s}s timeout=${gpu_wait_max_s}s"

# ---------------------------------------------------------------- phase 1: reuse
log "phase 1: reuse remaining Mtb Fold2/Fold3 features"
(
  cd "$workroot/code" || exit 1
  MTB_REUSE_DATASETS="Mtb_ethionamide_3fold:ethionamide Mtb_amoxicillin_3fold:amoxicillin" \
    bash "$scriptdir/run-mtb-fold23-fold1-reuse-v2.sh"
) > "$logdir/reuse.log" 2>&1
reuse_rc=$?
log "reuse finished (rc=$reuse_rc); see $logdir/reuse.log"

# ------------------------------------------------------------- phase 2: train/predict
train_one() {
    dataset=$1; fold=$2; modelbase_root=$3
    fdir="$featureroot/$dataset/$fold"
    base="$modelbase_root/$dataset/$fold"
    model="$base/model-gpu1"
    pred="$base/prediction-gpu1"
    mkdir -p "$base" "$logdir"

    if [ ! -f "$fdir/.build-train.complete" ] || [ ! -f "$fdir/.build-val.complete" ]; then
        log "SKIP train $dataset/$fold: features incomplete"
        return 1
    fi

    if [ -f "$model/models/best_model_f1_score.pt" ]; then
        log "SKIP train $dataset/$fold: model exists"
    else
        wait_for_gpu "train $dataset/$fold"
        log "train start $dataset/$fold on GPU$gpu"
        CUDA_VISIBLE_DEVICES=$gpu strainamr train \
            -i "$fdir/db" \
            --val-dir "$fdir/val_features" \
            -o "$model" \
            -s 1 --patience 20 --epochs 100 \
            > "$logdir/${dataset}-${fold}-train.log" 2>&1
        if [ $? -ne 0 ] || [ ! -f "$model/models/best_model_f1_score.pt" ]; then
            log "train FAILED $dataset/$fold (see $logdir/${dataset}-${fold}-train.log)"
            return 1
        fi
        log "train done $dataset/$fold"
    fi

    if [ -f "$pred/logs/output_sample_prob_predict.txt" ]; then
        log "SKIP predict $dataset/$fold: output exists"
        return 0
    fi
    wait_for_gpu "predict $dataset/$fold"
    log "predict start $dataset/$fold on GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu strainamr predict \
        -i "$fdir/val_features" \
        -m "$model" \
        -o "$pred" \
        --db "$fdir/db" \
        > "$logdir/${dataset}-${fold}-predict.log" 2>&1
    if [ $? -ne 0 ] || [ ! -f "$pred/logs/output_sample_prob_predict.txt" ]; then
        log "predict FAILED $dataset/$fold (see $logdir/${dataset}-${fold}-predict.log)"
        return 1
    fi
    log "predict done $dataset/$fold"
    return 0
}

status=0
log "phase 2: training + prediction"
for target in \
    "Mtb_ethionamide_3fold Fold2 models-3fold" \
    "Mtb_ethionamide_3fold Fold3 models-3fold" \
    "Mtb_amoxicillin_3fold Fold2 models-3fold" \
    "Mtb_amoxicillin_3fold Fold3 models-3fold" \
    "Sau_clindamycin_3fold Fold3 models-3fold" \
    "Sau_levofloxacin_3fold Fold2 models-3fold" \
    "Sau_levofloxacin_3fold Fold3 models-3fold"
do
    set -- $target
    train_one "$1" "$2" "$workroot/$3" || status=1
done

# --------------------------------------- phase 3: complete predictions that already have models
log "phase 3: prediction-only for Mtb ciprofloxacin/ofloxacin Fold2/3"
for target in \
    "Mtb_ciprofloxacin_3fold Fold2 models-baseline-extra-3fold" \
    "Mtb_ciprofloxacin_3fold Fold3 models-baseline-extra-3fold" \
    "Mtb_ofloxacin_3fold Fold2 models-baseline-extra-3fold" \
    "Mtb_ofloxacin_3fold Fold3 models-baseline-extra-3fold"
do
    set -- $target
    dataset=$1; fold=$2; root=$3
    fdir="$featureroot/$dataset/$fold"
    base="$workroot/$root/$dataset/$fold"
    pred="$base/prediction-gpu1"
    if [ ! -f "$base/model-gpu1/models/best_model_f1_score.pt" ]; then
        log "SKIP predict $dataset/$fold: no model"
        continue
    fi
    if [ -f "$pred/logs/output_sample_prob_predict.txt" ]; then
        log "SKIP predict $dataset/$fold: output exists"
        continue
    fi
    wait_for_gpu "predict $dataset/$fold"
    log "predict start $dataset/$fold on GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu strainamr predict \
        -i "$fdir/val_features" \
        -m "$base/model-gpu1" \
        -o "$pred" \
        --db "$fdir/db" \
        > "$logdir/${dataset}-${fold}-predict.log" 2>&1
    if [ $? -ne 0 ]; then
        log "predict FAILED $dataset/$fold"
        status=1
    else
        log "predict done $dataset/$fold"
    fi
done

if [ "$status" -eq 0 ]; then
    log "ALL DONE"
    touch "$rundir/.queue.complete"
else
    log "FINISHED WITH FAILURES (see logs in $logdir)"
fi
exit "$status"
