# StrainAMR

StrainAMR is a learning-based framework for predicting antimicrobial resistance (AMR) from
bacterial genome assemblies while exposing the genetic features that drive a call. Each genome is
encoded in three channels: 1. variation-graph SNV tokens, 2. protein-cluster (PC) tokens and 3. selected
k-mers, and scored by a multi-modal Transformer. A rule layer built from AMRFinderPlus and CARD
determinants can be combined with the model output, so every prediction comes with the evidence
behind it.

## Key Capabilities

- **AMR prediction from FASTA assemblies** -- one command per step, no manual feature engineering
- **Biologically interpretable feature discovery** -- attention weights, integrated gradients,
  occlusion and SHAP interaction values on all three channels
- **Token-to-feature mapping** -- every SNV, protein-cluster and k-mer token can be traced back to
  a gene or a genomic position
- **Known-mechanism layer** -- AMRFinderPlus/CARD determinants calibrated against the training
  labels, reported alongside the model probability
- **Three decision modes** -- hybrid, model-only and mechanism-only, chosen per analysis
- **Position-specific SNV encoding for *M. tuberculosis*** -- Scheme A, see `docs/mtb-schemea.md`
- **Parallel genome processing** -- `--threads` controls gene calling, RGI and feature extraction
- **Prebuilt models** -- one checkpoint per species-drug combination, see `models/manifest.tsv`

## Installation (Linux)

```bash
git clone https://github.com/liaoherui/StrainAMR.git
cd StrainAMR
unzip localDB.zip          # CARD database used by `rgi --local`
```

The exact package set validated for this release:

```bash
conda create -n strainamr --file docs/environment-explicit.txt
conda activate strainamr
pip install -r requirements.txt
pip install -e .
amrfinder -u  
```

`docs/environment-explicit.txt` pins every package to a version, build and hash, so conda installs
them as listed instead of solving dependencies. This is the recommended path.

`mamba` and `micromamba` take the same arguments and are usually faster:

```bash
mamba create -n strainamr --file docs/environment-explicit.txt
micromamba create -n strainamr --file docs/environment-explicit.txt
```

They are not part of a plain conda install. Install once with
`conda install -n base -c conda-forge mamba`, or use the micromamba install script. Plain conda
needs no extra step.

`environment.yml` describes the same environment with readable version pins and is kept for
anyone who wants to adjust it. It may fail to solve: bioconda's current builds of `rgi` and
`graphaligner` no longer resolve together (`rgi` pins `samtools 1.9`, which needs `zlib < 1.3` and
`ncurses < 5.10`). If that happens, use the explicit file above.

### External tools and databases

The pipeline also calls tools that pip cannot install: `prodigal`, `rgi` (CARD), `mmseqs` (or
`cd-hit`), `minimap2`, `seqwish`, `mash`, and for the mechanism layer `amrfinder`. All of them come
from the two conda channels above.

Two data assets have to be prepared before the first run: one is unpacked from this repository,
the other is downloaded.

```bash
unzip -q localDB.zip      # CARD database used by `rgi --local`
amrfinder -u              # AMRFinderPlus database, about 230 MB
```

## Troubleshooting

<details>

<summary><strong> Known-mechanism layer cannot run: amrfinder installed but no database found </strong></summary>

The AMRFinderPlus database is downloaded per environment, not shipped with the package. Run
`amrfinder -u` inside the same environment, or reuse an existing copy by pointing
`--amrfinder-db` at it.

</details>

<details>
<summary><strong> KeyError or card.json not found during build-train </strong></summary>

`localDB.zip` has not been unpacked. Run `unzip -q localDB.zip` in the repository root, or pass
`--rgi-extra <path>/localDB` to `rgi main`.
</details>

<details>

<summary><strong> seqwish exits with Illegal instruction </strong></summary>

The bioconda `seqwish` binary is compiled with AVX2. On older CPUs it crashes and no files appear
under `<output>/GFA_train_Minimap2/`. Run the job on a newer partition, or pass `-s 1 -p 1` to
`build-train` to skip the SNV and PC channels when you already have them.

</details>


<details>
<summary><strong>  Training runs on the CPU even though a GPU is present </strong></summary>

Possible reason: The conda `pytorch` package is a CPU build. On Ampere or newer cards (A6000, A100, RTX 30/40)
install the matching CUDA build, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu118`. A warning about `sm_86` in the log means the installed
build supports only older architectures.
</details>

<details>
<summary><strong>  Two or more GPUs are visible and training aborts with NCCL Error </strong></summary>

Restrict the run to one card with `CUDA_VISIBLE_DEVICES=0` (or whichever is free).
</details>

## Inputs

* **Genomes** -- one FASTA file per genome (`.fna`, `.fa` or `.fasta`) in a single directory.
* **Labels** -- a tab-separated file with a header line and two columns:

      Strain_ID	Label
      GCF_900126375	1
      GCF_900127785	0

  `Label` is `1` for resistant and `0` for susceptible. `Strain_ID` must match the FASTA file
  name without its extension. Genomes without a label are skipped with a warning, so a directory
  may contain more genomes than the label file names.

## Quick Start

Check the command line interface:

```bash
strainamr --help
strainamr build-train --help
```

Run the bundled demo (61 *S. aureus* genomes, 50 labelled):

```bash
unzip -q Test_genomes.zip
bash test_run.sh
```

Run the four steps on your own data:

```bash
# 1. features for the training genomes; the output directory becomes the training database
strainamr build-train -i genomes_train/ -l labels_train.txt -d imipenem -o db/ -t 32 \
    --amr-organism Klebsiella_pneumoniae

# 2. features for the test genomes against that database
strainamr build-test -i genomes_test/ -l labels_test.txt -d imipenem -o test_features/ \
    --db db/ -t 32

# 3. train; the validation features decide which checkpoint is kept
strainamr train -i db/ --val-dir test_features/ -o model/ -s 1 --patience 20 --epochs 100

# 4. predict
strainamr predict -i test_features/ -m model/ -o pred/ --db db/

# 5. optional: attribution and interaction analysis
strainamr interpret -i test_features/ -m model/ -o pred/analysis/
```

Steps 1 and 2 dominate the runtime because every genome goes through RGI. Step 3 is minutes on a
GPU; step 4 takes seconds unless attribution runs.

## Command-line Parameters

### `strainamr build-train`

Calls genes, annotates them with CARD/RGI, screens them with AMRFinderPlus, calibrates the
mechanism layer, builds the three token channels and selects features. The `-o` directory becomes
the training database every later step reads.

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Directory of training genome FASTA files |
| `-l`, `--label_file` | required | Phenotype label file |
| `-d`, `--drug` | required | Drug name to model |
| `-o`, `--outdir` | `StrainAMR_res` | Output directory, i.e. the training database |
| `-t`, `--threads` | `1` | Number of parallel worker processes |
| `-p`, `--pc` | `0` | Set to `1` to skip protein-cluster token generation |
| `-s`, `--snv` | `0` | Set to `1` to skip SNV token generation |
| `-k`, `--kmer` | `0` | Set to `1` to skip k-mer token generation |
| `--amr-organism` | none | AMRFinderPlus organism, e.g. `Klebsiella_pneumoniae`; required for the mechanism layer |
| `--gene-caller` | `pyrodigal` | `pyrodigal` (in-process) or `prodigal` (external binary) |
| `--cluster-tool` | `mmseqs` | `mmseqs` or `cd-hit` for protein clustering |
| `--no-mechanism` | off | Skip AMRFinderPlus screening and calibration |
| `--mech-min-carriers` | `5` | Training genomes that must carry a determinant before it is counted |
| `--mech-min-ppv` | `0.8` | Minimum PPV for a determinant to be counted |

### `strainamr build-test`

Repeats the encoding for new genomes, reusing the training database for vocabularies, protein
clusters and mechanism calibration.

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Directory of test genome FASTA files |
| `-l`, `--label_file` | optional | Test labels; placeholder labels are generated when omitted |
| `-d`, `--drug` | required | Drug name; must match the training data |
| `-o`, `--outdir` | required | Output directory for the test features |
| `--db` | output directory | Training database written by `build-train` |
| `-t`, `--threads` | `1` | Number of parallel worker processes |
| `-p`, `--pc` | `0` | Set to `1` to skip protein-cluster tokens |
| `-s`, `--snv` | `0` | Set to `1` to skip SNV tokens |
| `-k`, `--kmer` | `0` | Set to `1` to skip k-mer tokens |

### `strainamr train`

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Training database from `build-train` |
| `--val-dir` | none | Output directory of `build-test` |
| `-o`, `--outdir` | `StrainAMR_fold_res` | Directory for checkpoints, logs and SHAP output |
| `-s`, `--save_mode` | `1` | `1` keeps the best validation score, `0` the lowest validation loss |
| `--patience` | `20` | Epochs without improvement before early stopping |
| `--epochs` | `100` | Maximum number of epochs |
| `--lr` | `0.001` | Adam learning rate |
| `--batch_size` | `20` | Batch size for training and evaluation |
| `--head` | `flatten` | `flatten` (original) or `pooled` attention head |
| `-f`, `--feature_used` | `all` | Comma-separated list of channels (`kmer`, `snv`, `pc`) |
| `-t`, `--train_mode` | `0` | Set to `1` when only training data are available; creates a stratified split |
| `--val_ratio` | `0.2` | Validation fraction used with `-t 1` |

### `strainamr predict`

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Feature directory from `build-test` |
| `-m`, `--model_PATH` | required | Model directory (or a `.pt` file); `selected_model.txt` is used automatically |
| `-o`, `--outdir` | `StrainAMR_fold_res` | Prediction output directory |
| `--db` | input directory | Training database for token maps and mechanism calibration |
| `--mode` | `hybrid` | `hybrid`, `model-only` or `mechanism-only` |
| `--skip-analysis` | off | Skip SHAP and attribution, which take most of the runtime |
| `--batch_size` | `20` | Batch size for prediction |
| `--shap-top-k` | `10` | Top tokens written per sample for SHAP output |

### `strainamr interpret`

Runs only the attribution half on an existing checkpoint: `-i` feature directory, `-m` model
directory, `-o` output directory. Use it when predictions already exist and only the explanations
are needed, or to explain a different checkpoint.

## Output

**Feature extraction** (`build-train` / `build-test`)

- Token files: `strains_*_sentence_fs.txt` (SNV), `strains_*_pc_token_fs.txt` (protein cluster),
  `strains_*_kmer_token.txt` (k-mer), plus the unfiltered versions without `_fs`
- Mapping files: `node_token_match.txt`, `kmer_token_id.txt`, `feature_remain_graph.txt`,
  `feature_remain_pc.txt`
- `longest_len.txt`, `longest_len_fs.txt` -- padding widths per channel
- `merged_proteins_train.fa`, `merge_train_cdhit.clstr` -- training proteins and clusters, needed
  by `build-test`
- `mechanism/` -- `amrfinder_hits.tsv.gz`, `determinant_table.tsv`, `calibration.json`
- `build_params.json` -- gene caller, k-mer tool and mechanism settings recorded for later steps

**Model training** (`train`)

- `models/best_model_f1_score.pt` -- the selected checkpoint
- `selected_model.txt` -- relative path of that checkpoint, read by `predict`
- `token_lengths.txt` -- padding widths the checkpoint expects
- `mechanism_threshold.json` -- calibrated `tau` and the validation metrics behind it
- `logs/train_pred_log.txt` -- per-epoch loss and metrics
- `shap/`, `analysis/` -- feature rankings and attribution computed on the validation set

**Prediction** (`predict`)

- `predictions.tsv` -- one row per genome, see below
- `mechanism_evidence.tsv` -- every AMRFinderPlus / CARD hit with its training support
- `mechanism_summary.tsv` -- F1, sensitivity, specificity, precision, accuracy, AUC and the
  confusion matrix for `model@0.5`, `mechanism` and `hybrid@tau`
- `logs/output_sample_prob_predict.txt` -- raw probability per genome
- `analysis/transformer_attribution/` -- token attribution, deletion faithfulness, pair interactions
- `shap/` -- per-token SHAP values

### Reading `predictions.tsv`

| Column | Meaning |
| --- | --- |
| `Sample_ID` | Genome identifier |
| `Label` | Ground truth, `1` resistant / `0` susceptible, or `NA` when unknown |
| `Call` | Final call, `R` or `S` |
| `Basis` | Which evidence produced `Call`: `mechanism+model`, `model`, `mechanism`, or `none` |
| `Model_prob` | Model probability of resistance |
| `Model_call` | Call at the fixed 0.5 threshold |
| `Mechanism_call` | Call from the rule layer alone |
| `Determinants` | Counted determinants, e.g. `gyrA_S84L[supported]` |
| `Not_counted` | Hits that were found but not counted |
| `Flags` | `no_known_determinant`, `novel_to_training`, `model_disagrees` |

`Mechanism_call = none` means *no counted known determinant*, not *susceptible*. A determinant is
counted only when the training genomes support it; a contradicted or rare hit still appears in
`mechanism_evidence.tsv` but does not change the call.

## Prediction Modes

| Mode | Rule |
| --- | --- |
| `hybrid` (default) | resistant if the mechanism layer calls R **or** the model probability is at least tau |
| `model-only` | resistant if the model probability is at least 0.5 |
| `mechanism-only` | resistant if a counted known determinant is present; needs no model |

`tau` is not a fixed constant. It is chosen per model on the validation genomes to maximise the F1
of the hybrid rule, and stored in `mechanism_threshold.json` next to the checkpoint. Without that
file the pipeline falls back to `tau = 0.5` and prints a warning.

## Demo

`Test_genomes.zip` holds a small *S. aureus* / levofloxacin set: 61 genomes, 50 of which carry a
label and are used. The five validation genomes only show that the pipeline runs end to end, not
how well it performs.

```bash
bash test_run.sh
```

It unpacks the demo genomes, then runs build-train -> build-test -> train -> predict into
`Test_run/` and `Model_test/`. Steps that already finished are skipped, so re-running is cheap. One
GPU is used by default; override with `CUDA_VISIBLE_DEVICES`.

## Models

Trained weights are not stored in this repository; one checkpoint is 105-770 MB. They are published
separately, and `models/manifest.tsv` lists the best fold per species-drug combination together
with its validation AUC, F1, tau, SHA256 checksum and Google Drive file ID.

Download only the species-drug bundle needed for an analysis. Give the species as its full name or
use the corresponding short alias:

| Alias | Species |
| --- | --- |
| `Aba` | *Acinetobacter baumannii* |
| `Cdi` | *Clostridioides difficile* |
| `Ecoli` | *Escherichia coli* |
| `Kcp` | *Klebsiella pneumoniae* |
| `Mtb` | *Mycobacterium tuberculosis* |
| `Sau` | *Staphylococcus aureus* |

```bash
# show the 24 available combinations without downloading anything
strainamr download-model --list

# download, verify and extract only the S. aureus / erythromycin bundle
strainamr download-model -s Sau -d erythromycin -o pretrained_models/
```

The example writes the verified archive to `pretrained_models/archives/` and extracts the model to
`pretrained_models/Sau_erythromycin/`. A checksum mismatch stops the command and removes the
incomplete download. Omitting `-s` or `-d` is an error, so the command never downloads every model
by accident. Download all 24 bundles only when that is explicitly intended:

```bash
strainamr download-model --all -o pretrained_models/
```

Each drug below also links directly to its individual ZIP for browser download.

| Species | Drugs |
| --- | --- |
| *Acinetobacter baumannii* | [amikacin](https://drive.google.com/file/d/1uBpX39fcR3XLsd78iK6wX1yWA2yp-1Bd/view?usp=sharing), [ampicillin](https://drive.google.com/file/d/1aWajx9PH8uj0iw5B2fn-At1KpBe2ZKwF/view?usp=sharing), [avycaz](https://drive.google.com/file/d/1hhxUJv7s_cKl_oU5kh3LKsTqiyteZ-oL/view?usp=sharing), [imipenem](https://drive.google.com/file/d/1UpkiNLBB_JcBdUgpbH6jImNfe_syD4kz/view?usp=sharing) |
| *Clostridioides difficile* | [azithromycin](https://drive.google.com/file/d/1t3Zyut7kV9BqTmot3uo7Ls86xz36Tl4l/view?usp=sharing), [clarithromycin](https://drive.google.com/file/d/1y6OWopTe5Ojph1q_h1nHxJRhrBciz3OB/view?usp=sharing), [clindamycin](https://drive.google.com/file/d/1b9dflHt5IlJ10fJfa7gU5WMEt_7qL5Rt/view?usp=sharing), [moxifloxacin](https://drive.google.com/file/d/1WtY27bWw9Ykh0g5gdXxRTNz-QYR_POkT/view?usp=sharing) |
| *Escherichia coli* | [amoxicillin](https://drive.google.com/file/d/1bX0ap4h9sLjMoxphMhcRlN381eHKjwK0/view?usp=sharing), [ampicillin](https://drive.google.com/file/d/1RcUi6PILzo4AuQWxB89mab7r50w8RFFc/view?usp=sharing), [ciprofloxacin](https://drive.google.com/file/d/17nHo5wGP_5NCgs50Shj0mgsTycDa0aEL/view?usp=sharing), [levofloxacin](https://drive.google.com/file/d/1ptKWlXBRYiBvJzcK9LxdTTycW5F8tgzn/view?usp=sharing) |
| *Klebsiella pneumoniae* | [amoxicillin](https://drive.google.com/file/d/1Aiys5YQtazteb-nH6bh639TeOuThMnb2/view?usp=sharing), [avycaz](https://drive.google.com/file/d/1jIvcGYVCEvyB36ULknMsSpDFrdF0yZnG/view?usp=sharing), [ciprofloxacin](https://drive.google.com/file/d/1H7EejPH6YCPPjROVgNICDYzkEGXJHTOe/view?usp=sharing), [levofloxacin](https://drive.google.com/file/d/1JY5VW52IcHtirdbFTV2UTdjUAV4sFgFE/view?usp=sharing) |
| *Mycobacterium tuberculosis* | [amoxicillin](https://drive.google.com/file/d/103EbmrnS83deY8_3HasXbtCvXHY-ipWd/view?usp=sharing), [ciprofloxacin](https://drive.google.com/file/d/1jZRE6t2WPRJv6CR6aeVLJznTk-rUen9-/view?usp=sharing), [ethionamide](https://drive.google.com/file/d/1X3dZ3nsKWYTS-Wetwxs7LVxM_VnqweAm/view?usp=sharing), [ofloxacin](https://drive.google.com/file/d/1NWdF4ePRDudjvsIySbmc3l4tQJR5tXjI/view?usp=sharing) |
| *Staphylococcus aureus* | [ciprofloxacin](https://drive.google.com/file/d/1zMwlST9UpfMyqXBvw4KDIhGIx5liXX3A/view?usp=sharing), [clindamycin](https://drive.google.com/file/d/1AZLIv3lFFOR60HoaUB8YRDMCD3tsTyhO/view?usp=sharing), [erythromycin](https://drive.google.com/file/d/1mfmtMyo11NrN-NvtlHC117VRSOziB3T7/view?usp=sharing), [levofloxacin](https://drive.google.com/file/d/1Y1MASnuwJdXhP65Wn-hdSYNEdFZ32K9w/view?usp=sharing) |

[Browse or download all prebuilt model bundles in Google Drive](https://drive.google.com/drive/folders/1g633eWz1TKw5fnZ20k9BJ0MYZoL6zFQ2?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto).

To predict new genomes with the downloaded bundle, run from the StrainAMR repository root (where
`localDB/` was unpacked):

```bash
MODEL=pretrained_models/Sau_erythromycin
strainamr build-test -i genomes_test/ -d erythromycin -o "$MODEL/test_features/" \
    --db "$MODEL/db/" -t 32
strainamr predict -i "$MODEL/test_features/" -m "$MODEL/" -o "$MODEL/pred/" \
    --db "$MODEL/db/"
```

Add `-l labels_test.txt` to `build-test` when phenotype labels are available. Without labels the
pipeline still predicts every genome, but evaluation metrics are skipped. Use uncompressed FASTA
inputs (`.fna`, `.fa` or `.fasta`); do not keep a compressed copy of the same genome in the input
directory.

## Results

24 species-drug combinations, 6 species, three-fold cross-validation on the BV-BRC benchmark.
Mean F1 over all combinations is 0.917 (0.940 excluding Mtb-amoxicillin); 23 of 24 combinations
reproduce previously reported F1 within 0.03.

Two caveats when reading `models/manifest.tsv`. The listed F1 comes from the validation fold that
was used to select that fold, so it is optimistic; report the three-fold mean instead. And
*M. tuberculosis* amoxicillin remains the weakest combination (F1 0.393 at threshold 0.5) because
only 25 of 766 genomes are resistant.

## Repository Layout

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

If you use StrainAMR in your research, please cite:

> Liao et al. *StrainAMR: ...* (2025)

## License

To be decided by the maintainers; no license file is included yet.
