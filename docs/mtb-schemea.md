# Mtb Scheme A: position-specific SNV encoding

*M. tuberculosis* carries almost no accessory genome: resistance is driven by point mutations in
a handful of loci rather than by acquired genes. The generic variation-graph SNV channel does not
capture those mutations well, so Scheme A replaces **only the SNV channel** with tokens derived
from a fixed H37Rv region. The PC and k-mer channels are reused from the baseline fold.

## Region

For `ofloxacin` and `ciprofloxacin` the region is the `gyrBA` interval used by MTB-CNN:

    NC_000962.3:4997-9818     (0-based, half-open; 4,821 bp)

Use the H37Rv assembly `GCF_000195955.2`. Do not substitute another accession without checking
the coordinates first.

## Encoding

Each genome is aligned with `minimap2 -x asm5 -c --secondary=no`; alignments below MAPQ 20 or
200 bp are dropped. SNPs and indels are read from the CIGAR string and mapped to eleven stable
token ids per reference position: four SNP alleles, three insertion length bins (1, 2, >=3) and
three deletion length bins. Because the ids depend only on H37Rv position and variant class,
training and validation share one vocabulary, and token 0 remains the padding token.

This is a compact variant-token encoder, not a reproduction of the MTB-CNN sequence CNN: token
identity does not carry multi-base sequence context.

## Usage

    python scripts/mtb_schemea_features_v2.py \
      --fold-dir features-3fold/Mtb_ofloxacin_3fold/Fold1 \
      --out-dir  features-mtb-schemeA/Mtb_ofloxacin_3fold/Fold1 \
      --reference references/Mtb/H37Rv_NC_000962.3.fna \
      --drug ofloxacin \
      --code-dir . \
      --copy-baseline

`--code-dir` must point at this repository; the default inside the script is the development
path, so pass it explicitly. `--copy-baseline` reuses the existing PC and k-mer features instead
of rebuilding them. The script refuses to write into a non-empty `--out-dir`, so a failed run
cannot silently overwrite an earlier result.

Outputs are the new fold directory plus `mtb_feature_manifest.json` and `mtb_feature_qc.tsv`.

## Limitations

- Only `ofloxacin` and `ciprofloxacin` have a defined region. Amoxicillin would need a different
  locus (`blaC`, the PBPs) and has too few resistant genomes in the benchmark to fit.
- Genomes with no alignment to the region are rejected rather than silently encoded as
  all-reference. Check the QC table before training.
- A model trained on Scheme A features must be given Scheme A features at prediction time; the
  two SNV channels are not interchangeable.
