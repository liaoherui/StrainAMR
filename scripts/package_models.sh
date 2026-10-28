#!/usr/bin/env bash
# Build one release archive per species-drug combination from the training output tree.
#
#   bash scripts/package_models.sh <out_dir> [manifest]
#
# Each archive holds everything a user needs to run `predict` on the bundled validation features
# *and* `build-test` on new genomes: the checkpoint, the annotation settings, the token maps, the
# training proteins and clusters, the graphs new genomes align to, and the mechanism calibration.
#
# Set ONLY=Sau_levofloxacin_3fold to package a single dataset while testing.
set -euo pipefail

ROOT="${ROOT:-/data2/hmy/StrainAMR-optimize}"
MODELS="$ROOT/models-3fold"
FEATS="$ROOT/features-3fold"
OUT="${1:?usage: package_models.sh <out_dir> [manifest]}"
MANIFEST="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models/manifest.tsv}"

DB_FILES="node_token_match.txt kmer_token_id.txt feature_remain_graph.txt feature_remain_pc.txt longest_len.txt longest_len_fs.txt merged_proteins_train.fa merge_train_cdhit merge_train_cdhit.clstr duplicate_proteins_train.tsv pc_matches.txt build_params.json train_label.txt"
TRAIN_FEATS="strains_train_sentence.txt strains_train_sentence_fs.txt strains_train_pc_token.txt strains_train_pc_token_fs.txt strains_train_kmer_token.txt"
VAL_FEATS="strains_test_sentence.txt strains_test_sentence_fs.txt strains_test_pc_token.txt strains_test_pc_token_fs.txt strains_test_kmer_token.txt"
SHAP_REFS="strains_train_sentence_fs_shap_rmf.txt strains_train_pc_token_fs_shap_rmf.txt strains_train_kmer_token_shap_rmf.txt strains_train_sentence_fs_shap_filter.txt strains_train_pc_token_fs_shap_filter.txt strains_train_kmer_token_shap_filter.txt longest_len_fs_shap.txt"

mkdir -p "$OUT"

while read -r ds fold; do
  [ -n "$ds" ] || continue
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$ds" ]; then continue; fi
  mdir=$(ls -d "$MODELS/$ds/$fold"/model-gpu* 2>/dev/null | head -1 || true)
  db="$FEATS/$ds/$fold/db"
  val="$FEATS/$ds/$fold/val_features"
  [ -n "$mdir" ] || { echo "skip $ds/$fold: no model directory" >&2; continue; }
  [ -d "$db" ] || { echo "skip $ds/$fold: no feature database" >&2; continue; }

  name="${ds}__${fold}"
  archive="$OUT/$name.zip"
  if [ -f "$archive" ] && [ -s "$archive" ]; then echo "have    $name.zip"; continue; fi

  stage="$OUT/.stage-$name"
  rm -rf "$stage"
  mkdir -p "$stage/models" "$stage/db/shap" "$stage/db/mechanism" "$stage/val_features"

  cp "$mdir/models/best_model_f1_score.pt" "$stage/models/"
  cp "$mdir/selected_model.txt" "$mdir/token_lengths.txt" "$stage/"
  if [ -f "$mdir/mechanism_threshold.json" ]; then cp "$mdir/mechanism_threshold.json" "$stage/"; fi

  for f in $DB_FILES; do
    if [ -f "$db/$f" ]; then cp "$db/$f" "$stage/db/"; else echo "  MISSING db/$f" >&2; fi
  done
  for f in $TRAIN_FEATS; do
    if [ -f "$db/$f" ]; then cp "$db/$f" "$stage/db/"; else echo "  MISSING db/$f" >&2; fi
  done
  for f in $VAL_FEATS; do
    if [ -f "$val/$f" ]; then cp "$val/$f" "$stage/val_features/"; else echo "  MISSING val/$f" >&2; fi
  done
  for f in $SHAP_REFS; do
    if [ -f "$db/shap/$f" ]; then cp "$db/shap/$f" "$stage/db/shap/"; else echo "  MISSING db/shap/$f" >&2; fi
  done
  if [ -d "$db/mechanism" ]; then cp "$db/mechanism"/* "$stage/db/mechanism/" 2>/dev/null || true; fi
  if [ -d "$db/GFA_train_Minimap2" ]; then cp -r "$db/GFA_train_Minimap2" "$stage/db/"; fi
  for d in proteins_train rgi_train; do
    if [ -d "$db/$d" ]; then cp -r "$db/$d" "$stage/db/"; fi
  done
  for f in "$db"/kmer_selection_*.tsv; do
    if [ -f "$f" ]; then cp "$f" "$stage/db/"; fi
  done

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

if [ -f "$MANIFEST" ] && [ -z "${ONLY:-}" ]; then
  sha_table=$(mktemp)
  for z in "$OUT"/*.zip; do
    [ -f "$z" ] || continue
    printf "%s\t%s\n" "$(basename "$z")" "$(sha256sum "$z" | cut -d" " -f1)" >> "$sha_table"
  done
  awk -F"\t" -v OFS="\t" 'NR==FNR{h[$1]=$2;next} FNR==1{print;next} {if($10 in h) $11=h[$10]; print}' "$sha_table" "$MANIFEST" > "$MANIFEST.tmp"
  mv "$MANIFEST.tmp" "$MANIFEST"
  rm -f "$sha_table"
  echo "updated $MANIFEST"
fi

echo "archives: $(ls -1 "$OUT"/*.zip 2>/dev/null | wc -l)"
du -sh "$OUT"
