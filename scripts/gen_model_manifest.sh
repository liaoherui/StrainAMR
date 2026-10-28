#!/usr/bin/env bash
set -euo pipefail
out="${1:-models/manifest.tsv}"
mkdir -p "$(dirname "$out")"

species() {
  case "$1" in
    Aba)   echo "Acinetobacter baumannii" ;;
    Cdi)   echo "Clostridioides difficile" ;;
    Ecoli) echo "Escherichia coli" ;;
    Kcp)   echo "Klebsiella pneumoniae" ;;
    Mtb)   echo "Mycobacterium tuberculosis" ;;
    Sau)   echo "Staphylococcus aureus" ;;
    *)     echo "unknown" ;;
  esac
}

{
  printf 'dataset\tspecies\tdrug\tfold\tn_val\tauc\tf1_at_0.5\ttau\tmechanism\tfile\tsha256\n'
  while read -r ds fold n tau auc f1; do
    prefix=${ds%%_*}
    drug=${ds#*_}; drug=${drug%_3fold}
    mech=amrfinder
    [ "$prefix" = "Mtb" ] && mech=none
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s__%s.zip\t\n' \
      "$ds" "$(species "$prefix")" "$drug" "$fold" "$n" "$auc" "$f1" "$tau" "$mech" "$ds" "$fold"
  done <<'LIST'
Aba_amikacin_3fold        Fold1 86  0.4  0.8608 0.9313
Aba_ampicillin_3fold      Fold1 52  0.5  0.7964 0.8947
Aba_avycaz_3fold          Fold1 93  0.5  0.9964 0.9940
Aba_imipenem_3fold        Fold2 166 0.95 0.9385 0.9358
Cdi_azithromycin_3fold    Fold2 154 0.5  0.9915 0.9930
Cdi_clarithromycin_3fold  Fold1 154 0.5  1.0000 0.9930
Cdi_clindamycin_3fold     Fold2 121 0.5  1.0000 1.0000
Cdi_moxifloxacin_3fold    Fold2 154 0.7  0.9946 0.9844
Ecoli_amoxicillin_3fold   Fold1 365 0.5  0.9522 0.9296
Ecoli_ampicillin_3fold    Fold1 146 0.5  0.9698 0.9419
Ecoli_ciprofloxacin_3fold Fold1 507 0.5  0.9800 0.9741
Ecoli_levofloxacin_3fold  Fold1 136 0.5  0.9643 0.9794
Kcp_amoxicillin_3fold     Fold3 78  0.5  0.9901 0.9877
Kcp_avycaz_3fold          Fold1 83  0.5  0.9993 0.9787
Kcp_ciprofloxacin_3fold   Fold1 718 0.5  0.9888 0.9868
Kcp_levofloxacin_3fold    Fold1 608 0.5  0.9921 0.9865
Mtb_amoxicillin_3fold     Fold3 255 0.5  0.7667 0.6667
Mtb_ciprofloxacin_3fold   Fold3 112 0.5  1.0000 0.9600
Mtb_ethionamide_3fold     Fold1 188 0.5  0.7748 0.6621
Mtb_ofloxacin_3fold       Fold2 274 0.5  0.9362 0.9146
Sau_ciprofloxacin_3fold   Fold3 410 0.5  0.9924 0.9904
Sau_clindamycin_3fold     Fold3 208 0.5  0.9900 0.9746
Sau_erythromycin_3fold    Fold2 435 0.5  0.9900 0.9712
Sau_levofloxacin_3fold    Fold1 96  0.5  1.0000 1.0000
LIST
} > "$out"

printf 'wrote %s (%s lines)\n' "$out" "$(wc -l < "$out")"
