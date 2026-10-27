# StrainAMR

Antimicrobial resistance (AMR) prediction from bacterial genome assemblies.

Each genome is encoded in three channels — variation-graph SNV tokens, protein-cluster (PC)
tokens and selected k-mers — and scored by a multimodal Transformer. A rule layer built from
AMRFinderPlus / CARD determinants can be combined with the model output, so every prediction
comes with the evidence behind it.

Beyond prediction, StrainAMR identifies the AMR-related genomic features behind a call: which
SNV, protein-cluster and k-mer tokens drive an individual genome's probability, which features
are shared across resistant genomes, and which variant pairs act together. The same reports are
produced by `strainamr interpret` on an existing checkpoint, without retraining.

What is included:

- a single `strainamr` entry point for every step
- a known-mechanism layer calibrated on the training genomes (`mechanism_layer.py`)
- attribution outputs: token attribution, deletion faithfulness, pair interactions, SHAP
- a position-specific SNV encoder for *M. tuberculosis* (Scheme A, see `docs/mtb-schemea.md`)

## Install

The exact package set validated for this release:

    conda create -n strainamr --file docs/environment-explicit.txt
    conda activate strainamr
    pip install -r requirements.txt
    pip install -e .

`docs/environment-explicit.txt` pins all 240 packages to a version, build and hash, so conda
installs them as listed instead of solving dependencies. This is the recommended path.

`mamba` and `micromamba` accept the same arguments and are usually faster:

    mamba create -n strainamr --file docs/environment-explicit.txt
    micromamba create -n strainamr --file docs/environment-explicit.txt

They are not part of a plain conda install; install them once with
`conda install -n base -c conda-forge mamba`, or follow the micromamba install script. Plain
conda needs no extra step.

`environment.yml` describes the same environment with readable version pins and is kept for
anyone who wants to adjust it. It may fail to solve: bioconda's current builds of `rgi` and
`graphaligner` no longer resolve together (`rgi` pins `samtools 1.9`, which needs `zlib < 1.3`
and `ncurses < 5.10`). If that happens, use the explicit file above.

Two data assets have to be prepared before the first run: one is unpacked from this repository,
the other is downloaded.

    unzip -q localDB.zip      # CARD database used by `rgi --local`
    amrfinder -u              # AMRFinderPlus database, about 230 MB

The pipeline also calls tools that pip cannot install: `prodigal`, `rgi` (CARD), `mmseqs`
(or `cd-hit`), `minimap2`, `seqwish`, `mash`, and for the mechanism layer `amrfinder`.

## Inputs

* **Genomes** — one FASTA file per genome (`.fna`, `.fa` or `.fasta`) in a single directory.
* **Labels** — a tab-separated file with a header line and two columns:

      Strain_ID	Label
      GCF_900126375	1
      GCF_900127785	0

  `Label` is `1` for resistant and `0` for susceptible. `Strain_ID` must match the FASTA file
  name without its extension. Genomes without a label are skipped with a warning, so a directory
  may contain more genomes than the label file names.

## Quick start

    # 1. features for the training genomes; the output directory becomes the training database
    strainamr build-train -i genomes_train/ -l labels_train.txt -d imipenem -o db/ -t 32 \
        --amr-organism Klebsiella_pneumoniae

    # 2. features for the validation/test genomes against that database
    strainamr build-test -i genomes_test/ -l labels_test.txt -d imipenem -o test_features/ \
        --db db/ -t 32

    # 3. train; the validation features decide which checkpoint is kept
    strainamr train -i db/ --val-dir test_features/ -o model/ -s 1 --patience 20 --epochs 100

    # 4. predict (model + mechanism layer)
    strainamr predict -i test_features/ -m model/ -o pred/ --db db/

    # 5. optional: attribution and interaction analysis on the same data
    strainamr interpret -i test_features/ -m model/ -o pred/analysis/

Steps 1 and 2 are the expensive ones because they call RGI on every genome. Step 3 is minutes on
a GPU. Step 4 takes seconds unless the attribution step runs.

## Step reference

### `strainamr build-train`

Calls genes with pyrodigal, annotates them with CARD/RGI, screens them with AMRFinderPlus,
calibrates the mechanism layer on the training labels, then builds the three token channels and
selects features. Everything lands in the directory passed to `-o`, which is the training
database every later step reads.

| option | meaning |
|---|---|
| `-i` | directory of training genomes |
| `-l` | label file |
| `-d` | drug name; use the same string in every later step |
| `-o` | output directory, i.e. the training database |
| `-t` | number of processes |
| `--amr-organism` | AMRFinderPlus organism, e.g. `Klebsiella_pneumoniae`, `Staphylococcus_aureus`; required for the mechanism layer |
| `--gene-caller` | `pyrodigal` (default, in-process) or `prodigal` (external binary) |
| `--cluster-tool` | `mmseqs` (default) or `cd-hit` for protein clustering |
| `--no-mechanism` | skip AMRFinderPlus screening and calibration entirely |
| `--mech-min-carriers` | how many training genomes must carry a determinant before it is counted (default 5) |
| `--mech-min-ppv` | minimum PPV for a determinant to be counted (default 0.8) |

Outputs inside `-o`:

| path | content |
|---|---|
| `strains_train_sentence.txt` | variation-graph SNV tokens, one line per genome |
| `strains_train_pc_token.txt` | protein-cluster tokens |
| `strains_train_kmer_token.txt` | k-mer tokens |
| `*_fs.txt` | the same files after chi-square / SHAP feature selection; these are what training reads |
| `node_token_match.txt`, `kmer_token_id.txt` | token to gene / k-mer maps used by the attribution step |
| `feature_remain_graph.txt`, `feature_remain_pc.txt` | indices of the retained features |
| `longest_len.txt`, `longest_len_fs.txt` | padding widths per channel |
| `merged_proteins_train.fa`, `merge_train_cdhit.clstr` | training proteins and clusters; `build-test` needs them |
| `mechanism/` | `amrfinder_hits.tsv.gz`, `determinant_table.tsv`, `calibration.json` |
| `build_params.json` | gene caller, k-mer tool and mechanism settings recorded for later steps |

### `strainamr build-test`

Repeats the same encoding for new genomes, but reuses the training database for the vocabularies,
the protein clusters and the mechanism calibration. Nothing is re-trained.

| option | meaning |
|---|---|
| `-i`, `-l`, `-d`, `-t` | same meaning as in `build-train` |
| `-o` | output directory for the new features |
| `--db` | the training database written by `build-train` |

Outputs in `-o`: `strains_test_sentence_fs.txt`, `strains_test_pc_token_fs.txt`,
`strains_test_kmer_token.txt`, `Align_test_res/*.gaf` (graph alignments), `merge_test_cdhit*`,
and `mechanism/amrfinder_hits.tsv.gz` when the mechanism layer is active.

### `strainamr train`

Trains the multimodal Transformer on the training tokens and evaluates on the validation tokens
after every epoch.

| option | meaning |
|---|---|
| `-i` | training database from `build-train` |
| `--val-dir` | output directory of `build-test` |
| `-o` | model output directory |
| `-s` | `1` keeps the checkpoint with the best validation score, `0` the one with the lowest validation loss |
| `--patience` | stop after this many epochs without improvement |
| `--epochs` | maximum number of epochs |
| `--lr` | Adam learning rate (default 0.001) |
| `--head` | `flatten` (default) or `pooled` attention head |
| `--feature_used` | which channels to use, e.g. `all` or `snv,pc` |

Outputs in `-o`:

| path | content |
|---|---|
| `models/best_model_f1_score.pt` | the selected checkpoint |
| `selected_model.txt` | relative path of that checkpoint, read by `predict` |
| `token_lengths.txt` | padding widths the checkpoint expects |
| `mechanism_threshold.json` | calibrated `tau` and the validation metrics behind it |
| `logs/train_pred_log.txt` | per-epoch loss and metrics |
| `shap/`, `analysis/` | feature rankings and attribution computed on the validation set |

### `strainamr predict`

| option | meaning |
|---|---|
| `-i` | feature directory from `build-test` |
| `-m` | model directory (or a `.pt` file); `selected_model.txt` is used automatically |
| `-o` | prediction output directory |
| `--db` | training database, needed for token maps and mechanism calibration |
| `--mode` | `hybrid` (default), `model-only` or `mechanism-only` |
| `--skip-analysis` | skip SHAP and attribution, which take most of the runtime |

Outputs in `-o`:

| file | content |
|---|---|
| `predictions.tsv` | one row per genome, see below |
| `mechanism_evidence.tsv` | every AMRFinderPlus / CARD hit with its training support |
| `mechanism_summary.tsv` | F1, sensitivity, specificity, precision, accuracy, AUC and the confusion matrix for `model@0.5`, `mechanism` and `hybrid@tau` |
| `logs/output_sample_prob_predict.txt` | raw probability per genome |
| `analysis/transformer_attribution/` | token attribution, deletion faithfulness, pair interactions |
| `shap/` | per-token SHAP values |

### `strainamr interpret`

Runs only the attribution half on an existing checkpoint: `-i` feature directory, `-m` model
directory, `-o` output directory. Use it when you already have predictions and only want the
explanations, or when you want them for a different checkpoint.

## Reading predictions.tsv

| column | meaning |
|---|---|
| `Sample_ID` | genome identifier |
| `Label` | ground truth, `1` resistant / `0` susceptible, or `NA` when unknown |
| `Call` | final call, `R` or `S` |
| `Basis` | which evidence produced `Call`: `mechanism+model`, `model`, `mechanism`, or `none` |
| `Model_prob` | model probability of resistance |
| `Model_call` | call at the fixed 0.5 threshold |
| `Mechanism_call` | call from the rule layer alone |
| `Determinants` | counted determinants, e.g. `gyrA_S84L[supported]` |
| `Not_counted` | hits that were found but not counted |
| `Flags` | `no_known_determinant` (model called R with no counted determinant), `novel_to_training` (determinant absent from the training set), `model_disagrees` (model and mechanism disagree) |

`Mechanism_call = none` means *no counted known determinant*, not *susceptible*. A determinant is
counted only when the training genomes support it; a contradicted or rare hit still appears in
`mechanism_evidence.tsv` but does not change the call.

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

## Demo

`Test_genomes.zip` holds a small *S. aureus* / levofloxacin set: 61 genomes, 50 of which
carry a label and are used. The five validation genomes only show that the pipeline runs
end to end, not how well it performs.

The two data steps in Install are required first: `unzip -q localDB.zip` and `amrfinder -u`.
The demo runs the known-mechanism layer, which needs the AMRFinderPlus database.

    bash test_run.sh

It unpacks the demo genomes, then runs build-train -> build-test -> train -> predict into
`Test_run/` and `Model_test/`. Steps that already finished are skipped, so re-running is cheap.
One GPU is used by default; override with `CUDA_VISIBLE_DEVICES`.

## Models

Trained weights are not stored in this repository; one checkpoint is 105-770 MB. They are
published separately, and `models/manifest.tsv` lists the best fold per species-drug combination
together with its validation AUC, F1 and tau. `scripts/download_models.sh` fetches and verifies a
selection once the release is published.

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

    strainamr.py             command line entry point
    StrainAMR_*.py           the four pipeline steps plus attribution
    mechanism_layer.py       known-mechanism rule layer and its calibration
    library/                 Transformer, pooled head, attribution and SHAP helpers
    scripts/                 three-fold pipeline, Scheme A, threshold evaluation
    test_run.sh              end-to-end demo on the genomes in Test_genomes.zip
    requirements.txt         pip dependencies that conda does not provide
    docs/                    method notes and the explicit package list
    models/manifest.tsv      index of released checkpoints

## Citation

If you use StrainAMR, cite the original publication of the method (to be completed by the
maintainers).

## License

To be decided by the maintainers; no license file is included yet.
