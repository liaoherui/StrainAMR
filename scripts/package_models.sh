#!/usr/bin/env bash
# Build one release archive per species-drug combination from the training output tree.
#
#   bash scripts/package_models.sh <out_dir> [manifest]
#
# Each archive holds everything needed to run or retrain one fold: the checkpoint, the token
# maps and training proteins the predictor reads, the mechanism calibration, and the extracted
# train/validation feature files.
set -euo pipefail

ROOT="${ROOT:-/data2/hmy/StrainAMR-optimize}"
MODELS="$ROOT/models-3fold"
FEATS="$ROOT/features-3fold"
OUT="${1:?usage: package_models.sh <out_dir> [manifest]}"
MANIFEST="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models/manifest.tsv}"

MAPS="node_token_match.txt kmer_token_id.txt feature_remain_graph.txt feature_remain_pc.txt longest_len_fs.txt merged_proteins_train.fa merge_train_cdhit.clstr"
TRAIN_FEATS="strains_train_sentence.txt strains_train_sentence_fs.txt strains_train_pc_token.txt strains_train_pc_token_fs.txt strains_train_kmer_token.txt"
VAL_FEATS="strains_test_sentence.txt strains_test_sentence_fs.txt strains_test_pc_token.txt strains_test_pc_token_fs.txt strains_test_kmer_token.txt"

mkdir -p "$OUT"

while read -r ds fold; do
  [ -n "$ds" ] || continue
  mdir=$(ls -d "$MODELS/$ds/$fold"/model-gpu* 2>/dev/null | head -1 || true)
  db="$FEATS/$ds/$fold/db"
  val="$FEATS/$ds/$fold/val_features"
  if [ -z "$mdir" ]; then echo "skip $ds/$fold: no model directory" >&2; continue; fi
  if [ ! -d "$db" ]; then echo "skip $ds/$fold: no feature database" >&2; continue; fi
  if [ ! -f "$mdir/models/best_model_f1_score.pt" ]; then
    echo "skip $ds/$fold: checkpoint missing" >&2; continue
  fi

  name="${ds}__${fold}"
  archive="$OUT/$name.zip"
  if [ -f "$archive" ] && [ -s "$archive" ]; then
    echo "have    $name.zip"
    continue
  fi

  stage="$OUT/.stage-$name"
  rm -rf "$stage"
  mkdir -p "$stage/models" "$stage/db" "$stage/val_features"

  cp "$mdir/models/best_model_f1_score.pt" "$stage/models/"
  cp "$mdir/selected_model.txt" "$mdir/token_lengths.txt" "$stage/"
  if [ -f "$mdir/mechanism_threshold.json" ]; then
    cp "$mdir/mechanism_threshold.json" "$stage/"
  fi

  for f in $MAPS; do
    if [ -f "$db/$f" ]; then
      cp "$db/$f" "$stage/db/"
    else
      echo "  warning: $db/$f missing" >&2
    fi
  done

  for f in $TRAIN_FEATS; do
    if [ -f "$db/$f" ]; then
      cp "$db/$f" "$stage/db/"
    else
      echo "  warning: $db/$f missing" >&2
    fi
  done

  for f in $VAL_FEATS; do
    if [ -f "$val/$f" ]; then
      cp "$val/$f" "$stage/val_features/"
    else
      echo "  warning: $val/$f missing" >&2
    fi
  done

  if [ -d "$db/mechanism" ]; then
    mkdir -p "$stage/db/mechanism"
    cp "$db/mechanism"/* "$stage/db/mechanism/"
  fi

  ( cd "$stage" && zip -qr "$archive" . )
  rm -rf "$stage"
  echo "built   $name.zip  $(du -h "$archive" | cut -f1)"
done <<'LIST'
Aba_amikacin_3fold Fold1
Aba_ampicillin_3fold Fold1
Aba_avycaz_3fold Fold1
Aba_imipenem_3fold Fold2
Cdi_azithromycin_3fold Fold2
Cdi_clarithromycin_3fold Fold1
Cdi_clindamycin_3fold Fold2
Cdi_moxifloxacin_3fold Fold2
Ecoli_amoxicillin_3fold Fold1
Ecoli_ampicillin_3fold Fold1
Ecoli_ciprofloxacin_3fold Fold1
Ecoli_levofloxacin_3fold Fold1
Kcp_amoxicillin_3fold Fold3
Kcp_avycaz_3fold Fold1
Kcp_ciprofloxacin_3fold Fold1
Kcp_levofloxacin_3fold Fold1
Mtb_amoxicillin_3fold Fold3
Mtb_ciprofloxacin_3fold Fold3
Mtb_ethionamide_3fold Fold1
Mtb_ofloxacin_3fold Fold2
Sau_ciprofloxacin_3fold Fold3
Sau_clindamycin_3fold Fold3
Sau_erythromycin_3fold Fold2
Sau_levofloxacin_3fold Fold1
LIST

if [ -f "$MANIFEST" ]; then
  sha_table=$(mktemp)
  for z in "$OUT"/*.zip; do
    [ -f "$z" ] || continue
    printf '%s\t%s\n' "$(basename "$z")" "$(sha256sum "$z" | cut -d' ' -f1)" >> "$sha_table"
  done
  awk -F'\t' -v OFS='\t' '
    NR==FNR { h[$1]=$2; next }
    FNR==1  { print; next }
    { if ($10 in h) $11=h[$10]; print }
  ' "$sha_table" "$MANIFEST" > "$MANIFEST.tmp"
  mv "$MANIFEST.tmp" "$MANIFEST"
  rm -f "$sha_table"
  echo "updated $MANIFEST"
fi

echo "archives: $(ls -1 "$OUT"/*.zip 2>/dev/null | wc -l)"
du -sh "$OUT"
