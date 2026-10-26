#!/usr/bin/env bash
# Reuse Mtb Fold1 per-genome intermediates to build Fold2/Fold3 features.
#
# v2: the dataset list is configurable instead of hard-coded, so the remaining
# Mtb folds (ethionamide, amoxicillin) can be built without editing the script.
#
#   MTB_REUSE_DATASETS="Mtb_ethionamide_3fold:ethionamide" scripts/run-mtb-fold23-fold1-reuse-v2.sh
#
# Datasets run sequentially; each gets its own runner log. Existing markers are
# honoured, so re-running is safe and skips completed folds.
set -euo pipefail

workroot=/data2/hmy/StrainAMR-optimize
code="$workroot/code"
featureroot="$workroot/features-3fold"
manifestroot="$workroot/reproduction-manifests/fold-labels"
cacheroot="$workroot/sample-cache/Mtb_fold1_reuse"

prefill="$workroot/scripts/prefill-feature-cache.py"
makecache="$workroot/scripts/build-fold1-reuse-cache.py"

threads=32
organism=Mycobacterium_tuberculosis
amr_database=2023-11-15.1

if ! command -v strainamr >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source /home/lhr/miniconda3/etc/profile.d/conda.sh
    conda activate strainamr
    set -u
fi

command -v strainamr >/dev/null 2>&1 || {
    echo "strainamr not found; activate the strainamr environment first" >&2
    exit 1
}

cd "$code"

mkdir -p "$featureroot"
exec 9>"$featureroot/.mtb-fold23-reuse.lock"
flock -n 9 || {
    echo "another run already holds $featureroot/.mtb-fold23-reuse.lock" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

materialize_split() {
    manifest=$1
    genomedir=$2
    labelfile=$3

    mkdir -p "$genomedir" "$(dirname "$labelfile")"

    printf 'Strain_ID\tLabel\n' > "$labelfile"

    while IFS=$'\t' read -r strain_id label bvbrc_id genome_path status rest
    do
        strain_id=${strain_id//$'\r'/}
        label=${label//$'\r'/}
        genome_path=${genome_path//$'\r'/}
        status=${status//$'\r'/}

        [ "$strain_id" != "Strain_ID" ] || continue
        [ "$status" = "READY" ] || {
            echo "Not READY: $strain_id status=$status" >&2
            return 1
        }
        [ -s "$genome_path" ] || {
            echo "Missing genome: $genome_path" >&2
            return 1
        }

        destination="$genomedir/${strain_id}.fna"

        if [ -L "$destination" ]; then
            current_target=$(readlink -f "$destination")
            expected_target=$(readlink -f "$genome_path")
            [ "$current_target" = "$expected_target" ] || {
                echo "Conflicting link: $destination" >&2
                return 1
            }
        elif [ -e "$destination" ]; then
            echo "Destination exists and is not a link: $destination" >&2
            return 1
        else
            ln -s "$genome_path" "$destination"
        fi

        printf '%s\t%s\n' "$strain_id" "$label" >> "$labelfile"

    done < "$manifest"
}

run_fold() {
    dataset=$1
    fold=$2
    drug=$3
    cache=$4

    foldroot="$featureroot/$dataset/$fold"
    inputroot="$foldroot/input"
    logdir="$foldroot/logs"

    traingenomes="$inputroot/genomes-train"
    valgenomes="$inputroot/genomes-val"
    trainlabel="$inputroot/labels/train_label.tsv"
    vallabel="$inputroot/labels/val_label.tsv"

    dbdir="$foldroot/db"
    valfeatures="$foldroot/val_features"
    trainlog="$logdir/build-train.log"
    vallog="$logdir/build-val.log"
    trainmarker="$foldroot/.build-train.complete"
    valmarker="$foldroot/.build-val.complete"

    foldmanifest="$manifestroot/$dataset/$fold"

    mkdir -p "$logdir"

    log "===== $dataset $fold ($drug) ====="

    materialize_split \
        "$foldmanifest/train_full_manifest.tsv" \
        "$traingenomes" \
        "$trainlabel"
    materialize_split \
        "$foldmanifest/test_manifest.tsv" \
        "$valgenomes" \
        "$vallabel"

    train_count=$(( $(wc -l < "$trainlabel") - 1 ))
    val_count=$(( $(wc -l < "$vallabel") - 1 ))
    log "$fold samples: train=$train_count val=$val_count"

    if [ ! -f "$trainmarker" ]; then
        log "$fold prefill train from Fold1 cache (no AMRFinder)"
        python "$prefill" \
            --labels "$trainlabel" \
            --work-dir "$dbdir" \
            --sample-cache "$cache" \
            --amrfinder-cache "$cache/AMRFinderPlus/$amr_database" \
            --skip-amrfinder \
            --layout train \
            --organism "$organism" \
            --mode protein \
            --software-version 3.11.2 \
            --database-version "$amr_database"

        log "$fold build-train (--no-mechanism)"
        strainamr build-train \
            -i "$traingenomes" \
            -o "$dbdir" \
            -l "$trainlabel" \
            -d "$drug" \
            -t "$threads" \
            --no-kmer-weights \
            --cluster-tool cd-hit \
            --no-mechanism \
            > "$trainlog" 2>&1

        touch "$trainmarker"
        log "$fold build-train COMPLETE"
    else
        log "$fold build-train already complete"
    fi

    if [ ! -f "$valmarker" ]; then
        log "$fold prefill validation from Fold1 cache (no AMRFinder)"
        python "$prefill" \
            --labels "$vallabel" \
            --work-dir "$valfeatures" \
            --sample-cache "$cache" \
            --amrfinder-cache "$cache/AMRFinderPlus/$amr_database" \
            --skip-amrfinder \
            --layout test \
            --organism "$organism" \
            --mode protein \
            --software-version 3.11.2 \
            --database-version "$amr_database"

        log "$fold build-test"
        strainamr build-test \
            -i "$valgenomes" \
            -l "$vallabel" \
            -d "$drug" \
            -o "$valfeatures" \
            --db "$dbdir" \
            -t "$threads" \
            > "$vallog" 2>&1

        touch "$valmarker"
        log "$fold build-test COMPLETE"
    else
        log "$fold build-test already complete"
    fi

    log "$dataset $fold DONE"
}

run_dataset() {
    dataset=$1
    drug=$2
    cache="$cacheroot/$dataset"
    fold1dir="$featureroot/$dataset/Fold1"
    foldmanifest="$manifestroot/$dataset"

    [ -f "$fold1dir/.build-train.complete" ] || {
        echo "$dataset Fold1 build-train is incomplete" >&2
        exit 1
    }
    [ -f "$fold1dir/.build-val.complete" ] || {
        echo "$dataset Fold1 build-val is incomplete" >&2
        exit 1
    }

    log "building Fold1-derived cache for $dataset (Genes/Proteins/RGI only)"
    python "$makecache" \
        --fold1-dir "$fold1dir" \
        --cache-dir "$cache" \
        --manifests \
            "$foldmanifest/Fold2/train_full_manifest.tsv" \
            "$foldmanifest/Fold2/test_manifest.tsv" \
            "$foldmanifest/Fold3/train_full_manifest.tsv" \
            "$foldmanifest/Fold3/test_manifest.tsv" \
        --amrfinder-out "$cache/AMRFinderPlus/$amr_database" \
        --skip-amrfinder

    run_fold "$dataset" Fold2 "$drug" "$cache"
    run_fold "$dataset" Fold3 "$drug" "$cache"
}


datasets=${MTB_REUSE_DATASETS:-"Mtb_ofloxacin_3fold:ofloxacin Mtb_ciprofloxacin_3fold:ciprofloxacin"}
log "Mtb reuse run starting: $datasets"

status=0
for item in $datasets; do
    dataset=${item%%:*}
    drug=${item##*:}
    log "----- $dataset ($drug) -----"
    if run_dataset "$dataset" "$drug" > "$featureroot/$dataset-Fold23.runner.log" 2>&1; then
        log "$dataset reuse COMPLETE"
    else
        log "$dataset reuse FAILED (see $featureroot/$dataset-Fold23.runner.log)"
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    log "Mtb reuse run FAILED"
    exit 1
fi

log "verifying markers"
for item in $datasets; do
    dataset=${item%%:*}
    for fold in Fold2 Fold3; do
        foldroot="$featureroot/$dataset/$fold"
        for marker in .build-train.complete .build-val.complete; do
            if [ ! -f "$foldroot/$marker" ]; then
                echo "MISSING: $dataset/$fold/$marker" >&2
                status=1
            fi
        done
        printf '%-32s %-6s val_feature_entries=%s\n' \
            "$dataset" "$fold" \
            "$(find "$foldroot/val_features" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
    done
done

if [ "$status" -ne 0 ]; then
    log "Mtb reuse run FAILED"
    exit 1
fi

log "Mtb reuse run COMPLETE"
