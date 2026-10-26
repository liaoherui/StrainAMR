# StrainAMR

Antimicrobial resistance (AMR) prediction from bacterial genome assemblies.

Each genome is encoded in three channels — variation-graph SNV tokens, protein-cluster (PC)
tokens and selected k-mers — and scored by a multimodal Transformer. A rule layer built from
AMRFinderPlus / CARD determinants can be combined with the model output, so every prediction
comes with the evidence behind it.

This branch adds, on top of the original pipeline:

- a single `strainamr` entry point for every step
- a known-mechanism layer calibrated on the training genomes (`mechanism_layer.py`)
- attribution outputs: token attribution, deletion faithfulness, pair interactions, SHAP
- a position-specific SNV encoder for *M. tuberculosis* (Scheme A, see `docs/mtb-schemea.md`)

## Install

    conda env create -f environment.yml
    conda activate strainamr
    pip install -e .

The pipeline also calls tools that pip cannot install: `prodigal`, `rgi` (CARD), `mmseqs`
(or `cd-hit`), `minimap2`, `seqwish`, `mash`, and for the mechanism layer `amrfinder`
(`conda install -c bioconda ncbi-amrfinderplus && amrfinder -u`).

## Quick start

    strainamr build-train -i genomes_train/ -l labels.txt -d imipenem -o db/ -t 32
    strainamr train       -i db/ --val-dir val/db -o model/ -s 1 --patience 20 --epochs 100
    strainamr build-test  -i genomes_test/ -l labels_test.txt -d imipenem -o test/ --db db/ -t 32
    strainamr predict     -i test/ -m model/ -o pred/ --db db/
    strainamr interpret   -i test/ -m model/ -o pred/analysis/

`strainamr --help` lists all subcommands; `strainamr <subcommand> --help` lists that step's
options. The original standalone scripts still work unchanged.

## Prediction modes

`strainamr predict --mode` selects how the final call is made.

| mode | rule |
|---|---|
| `hybrid` (default) | resistant if the mechanism layer calls R **or** the model probability is at least tau |
| `model-only` | resistant if the model probability is at least 0.5 |
| `mechanism-only` | resistant if a counted known determinant is present; needs no model |

`tau` is not a fixed constant. It is chosen per model on the validation genomes to maximise the
F1 of the hybrid rule, and stored in `mechanism_threshold.json` next to the checkpoint. Without
that file the pipeline falls back to `tau = 0.5` and prints a warning.

`Mechanism_call = none` means *no counted known determinant*, not *susceptible*. A determinant is
counted only when the training genomes support it (`min_carriers = 5`, `min_ppv = 0.8`); a
contradicted or rare hit is still written to `mechanism_evidence.tsv` but does not change the call.

## Models

Trained weights are not stored in this repository; one checkpoint is 105-770 MB. They are
published as release assets. `models/manifest.tsv` lists the best fold per species-drug
combination together with its validation AUC, F1 and tau, and `scripts/download_models.sh`
fetches and verifies a selection.

| species | drugs |
|---|---|
| *Acinetobacter baumannii* | amikacin, ampicillin, avycaz, imipenem |
| *Clostridioides difficile* | azithromycin, clarithromycin, clindamycin, moxifloxacin |
| *Escherichia coli* | amoxicillin, ampicillin, ciprofloxacin, levofloxacin |
| *Klebsiella pneumoniae* | amoxicillin, avycaz, ciprofloxacin, levofloxacin |
| *Mycobacterium tuberculosis* | amoxicillin, ciprofloxacin, ethionamide, ofloxacin |
| *Staphylococcus aureus* | ciprofloxacin, clindamycin, erythromycin, levofloxacin |

## Results

24 species-drug combinations, 6 species, three-fold cross-validation on the BV-BRC benchmark.
Mean F1 over all combinations is 0.917 (0.940 excluding Mtb-amoxicillin); 23 of 24 combinations
reproduce previously reported F1 within 0.03.

Two caveats when reading `models/manifest.tsv`. The listed F1 comes from the validation fold that
was used to select that fold, so it is optimistic; report the three-fold mean instead. And
*M. tuberculosis* amoxicillin remains the weakest combination (F1 0.393 at threshold 0.5) because
only 25 of 766 genomes are resistant.

## Repository layout

    strainamr.py            command line entry point
    StrainAMR_*.py          the four pipeline steps plus attribution
    mechanism_layer.py      known-mechanism rule layer and its calibration
    library/                Transformer, pooled head, attribution and SHAP helpers
    scripts/                three-fold pipeline, Scheme A, threshold evaluation
    docs/                   method notes
    models/manifest.tsv     index of released checkpoints

## Citation

If you use StrainAMR, cite the original publication of the method (to be completed by the
maintainers).

## License

To be decided by the maintainers; no license file is included yet.
