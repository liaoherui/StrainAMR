#!/usr/bin/env bash
set -euo pipefail

workroot=/data2/hmy/StrainAMR-optimize
code="$workroot/code"
manifestroot="$workroot/reproduction-manifests/fold-labels"
featureroot="$workroot/features-3fold"

threads=32

prefillscript="$workroot/scripts/prefill-feature-cache.py"
ecoli_sample_cache="$workroot/sample-cache/E_coli"
ecoli_amr_cache="$ecoli_sample_cache/AMRFinderPlus/Escherichia-3.11.2-db-2023-11-15.1"
kcp_sample_cache="$workroot/sample-cache/K_pneumoniae"
kcp_amr_cache="$kcp_sample_cache/AMRFinderPlus/Klebsiella_pneumoniae-3.11.2-db-2023-11-15.1"

mkdir -p "$featureroot"
cd "$code"

get_drug_name () {
    case "$1" in
        avycaz)
            echo "ceftazidime-avibactam"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

get_organism () {
    case "$1" in
        Aba)   echo "Acinetobacter_baumannii" ;;
        Cdi)   echo "Clostridioides_difficile" ;;
        Ecoli) echo "Escherichia" ;;
        Kcp)   echo "Klebsiella_pneumoniae" ;;
        Mtb)   echo "Mycobacterium_tuberculosis" ;;
        Sau)   echo "Staphylococcus_aureus" ;;
        *)
            echo "Unknown species prefix: $1" >&2
            return 1
            ;;
    esac
}

materialize_split () {
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
            echo "Destination already exists and is not a link: $destination" >&2
            return 1
        else
            ln -s "$genome_path" "$destination"
        fi

        printf '%s\t%s\n' "$strain_id" "$label" >> "$labelfile"

    done < "$manifest"
}

summary="$featureroot/feature-build-status.tsv"

if [ ! -s "$summary" ]; then
    printf 'Dataset\tFold\tStage\tStatus\tTimestamp\n' > "$summary"
fi

for datasetdir in "$manifestroot"/*_3fold
do
    dataset=$(basename "$datasetdir")
    prefix=${dataset%%_*}

    dataset_drug=${dataset#*_}
    dataset_drug=${dataset_drug%_3fold}
    drug=$(get_drug_name "$dataset_drug")

    organism=$(get_organism "$prefix")
    if [ "$prefix" = "Mtb" ]; then
        mechanism_option=(--no-mechanism)
    else
        mechanism_option=(--amr-organism "$organism")
    fi

    for fold in Fold1 Fold2 Fold3
    do
        foldmanifest="$datasetdir/$fold"

        trainmanifest="$foldmanifest/train_full_manifest.tsv"
        valmanifest="$foldmanifest/test_manifest.tsv"

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

        mkdir -p "$logdir"

        echo "=================================================="
        echo "Dataset: $dataset"
        echo "Fold: $fold"
        echo "Drug: $drug"
        echo "Organism: $organism"
        echo "=================================================="

        materialize_split \
            "$trainmanifest" \
            "$traingenomes" \
            "$trainlabel"

        materialize_split \
            "$valmanifest" \
            "$valgenomes" \
            "$vallabel"

        train_count=$(( $(wc -l < "$trainlabel") - 1 ))
        val_count=$(( $(wc -l < "$vallabel") - 1 ))

        echo "Train samples: $train_count"
        echo "Validation samples: $val_count"

        if [ ! -f "$trainmarker" ]; then
            printf '%s\t%s\tbuild-train\tRUNNING\t%s\n' \
                "$dataset" "$fold" "$(date '+%F %T')" \
                >> "$summary"

            if [ "$prefix" = "Ecoli" ]; then
                echo "Prefilling E. coli training cache: $dataset/$fold"

                python "$prefillscript" \
                    --labels "$trainlabel" \
                    --work-dir "$dbdir" \
                    --sample-cache "$ecoli_sample_cache" \
                    --amrfinder-cache "$ecoli_amr_cache" \
                    --layout train \
                    --organism Escherichia \
                    --mode protein \
                    --software-version 3.11.2 \
                    --database-version 2023-11-15.1 \
                    --cache-prefix ecoli_
            fi

            if [ "$prefix" = "Kcp" ]; then
                echo "Prefilling K. pneumoniae training cache: $dataset/$fold"

                python "$prefillscript"                     --labels "$trainlabel"                     --work-dir "$dbdir"                     --sample-cache "$kcp_sample_cache"                     --amrfinder-cache "$kcp_amr_cache"                     --layout train                     --organism Klebsiella_pneumoniae                     --mode protein                     --software-version 3.11.2                     --database-version 2023-11-15.1
            fi

            strainamr build-train \
                -i "$traingenomes" \
                -o "$dbdir" \
                -l "$trainlabel" \
                -d "$drug" \
                -t "$threads" \
                --no-kmer-weights \
                --cluster-tool cd-hit \
                "${mechanism_option[@]}" \
                > "$trainlog" 2>&1

            touch "$trainmarker"

            printf '%s\t%s\tbuild-train\tCOMPLETE\t%s\n' \
                "$dataset" "$fold" "$(date '+%F %T')" \
                >> "$summary"
        else
            echo "Skipping completed build-train"
        fi

        if [ ! -f "$valmarker" ]; then
            printf '%s\t%s\tbuild-val\tRUNNING\t%s\n' \
                "$dataset" "$fold" "$(date '+%F %T')" \
                >> "$summary"

            if [ "$prefix" = "Ecoli" ]; then
                echo "Prefilling E. coli validation cache: $dataset/$fold"

                python "$prefillscript" \
                    --labels "$vallabel" \
                    --work-dir "$valfeatures" \
                    --sample-cache "$ecoli_sample_cache" \
                    --amrfinder-cache "$ecoli_amr_cache" \
                    --layout test \
                    --organism Escherichia \
                    --mode protein \
                    --software-version 3.11.2 \
                    --database-version 2023-11-15.1 \
                    --cache-prefix ecoli_
            fi

            if [ "$prefix" = "Kcp" ]; then
                echo "Prefilling K. pneumoniae validation cache: $dataset/$fold"

                python "$prefillscript"                     --labels "$vallabel"                     --work-dir "$valfeatures"                     --sample-cache "$kcp_sample_cache"                     --amrfinder-cache "$kcp_amr_cache"                     --layout test                     --organism Klebsiella_pneumoniae                     --mode protein                     --software-version 3.11.2                     --database-version 2023-11-15.1
            fi

            strainamr build-test \
                -i "$valgenomes" \
                -l "$vallabel" \
                -d "$drug" \
                -o "$valfeatures" \
                --db "$dbdir" \
                -t "$threads" \
                > "$vallog" 2>&1

            touch "$valmarker"

            printf '%s\t%s\tbuild-val\tCOMPLETE\t%s\n' \
                "$dataset" "$fold" "$(date '+%F %T')" \
                >> "$summary"
        else
            echo "Skipping completed build-val"
        fi

        echo "COMPLETED: $dataset $fold"
    done
done

echo "ALL 3-FOLD FEATURE EXTRACTIONS COMPLETED"
