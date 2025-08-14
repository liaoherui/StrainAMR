# StrainAMR

StrainAMR is a learning-based framework for predicting antimicrobial resistance (AMR) from bacterial genomes while exposing the genetic features that drive resistance. The toolkit combines k‑mers, single nucleotide variants (SNVs) and protein clusters to build an interpretable classifier and highlight meaningful feature pairs through attention and SHAP interaction scores.

## Key Capabilities

- **Accurate AMR prediction** for bacterial strains from raw FASTA assemblies
- **Biologically interpretable feature discovery** using attention weights and SHAP interaction values
- **Parallel genome processing** with configurable thread count
- **Token-to-feature mapping** to translate model inputs back to genes, k‑mers and SNVs

## Installation (Linux/Ubuntu)

```bash
git clone https://github.com/liaoherui/StrainAMR.git
cd StrainAMR
unzip Test_genomes.zip
unzip localDB.zip
unzip Benchmark_features.zip
```

Install the helper utility:

```bash
pip install gdown
```

### Create the Conda environment

**Option 1 — build from `strainamr.yaml`:**

```bash
conda env create -f strainamr.yaml
conda activate strainamr
```

**Option 2 — use the pre-built environment (recommended):**

```bash
sh download_env.sh
source strainamr/bin/activate
```

### PhenotypeSeeker environment

```bash
sh download_ps.sh
python install_rebuild_ps.py
```

Add PhenotypeSeeker to your `PATH` (replace `/path/to/StrainAMR` with your directory):

```bash
echo "export PATH=\$PATH:/path/to/StrainAMR/PhenotypeSeeker/.PSenv/bin" >> ~/.bashrc
source ~/.bashrc
```

## Quick Start

Check the command-line interfaces:

```bash
python StrainAMR_build_train.py -h
python StrainAMR_build_test.py -h
python StrainAMR_model_train.py -h
python StrainAMR_model_predict.py -h
```

Run the end‑to‑end demo on the bundled test genomes:

```bash
sh test_run.sh
```

Reproduce the three‑fold cross‑validation experiment from the paper:

```bash
sh batch_train_3fold_exp.sh
```

## Command-line Parameters

### `StrainAMR_build_train.py`

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Directory containing training genome FASTA files |
| `-l`, `--label_file` | required | Path to phenotype label file |
| `-d`, `--drug` | required | Drug name to model |
| `-p`, `--pc` | `0` | Skip protein-cluster token generation when set to `1` |
| `-s`, `--snv` | `0` | Skip SNV token generation when set to `1` |
| `-k`, `--kmer` | `0` | Skip k‑mer token generation when set to `1` |
| `-t`, `--threads` | `1` | Number of parallel worker processes |
| `-o`, `--outdir` | `StrainAMR_res` | Output directory for generated features |

### `StrainAMR_model_train.py`

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Directory produced by build scripts containing token files |
| `-f`, `--feature_used` | `all` | Comma-separated list of features to use (`kmer`, `snv`, `pc`) |
| `-t`, `--train_mode` | `0` | Set to `1` if only training data are provided |
| `-s`, `--save_mode` | `1` | `0` saves model with minimum validation loss |
| `-a`, `--attention_weight` | `1` | `0` disables saving attention matrices |
| `-o`, `--outdir` | `StrainAMR_fold_res` | Directory for models, logs and SHAP outputs |

### `StrainAMR_model_predict.py`

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input_file` | required | Directory of feature files for prediction |
| `-f`, `--feature_used` | `all` | Feature types to use (`kmer`, `snv`, `pc`) |
| `-m`, `--model_PATH` | required | Directory containing pre-trained models |
| `-o`, `--outdir` | `StrainAMR_fold_res` | Directory for logs, SHAP results and analysis outputs |

## Output

- **Feature extraction** (`StrainAMR_build_train.py` / `StrainAMR_build_test.py`)
  - Token files such as `strains_*_sentence_fs.txt`, `strains_*_pc_token_fs.txt`, `strains_*_kmer_token.txt`
  - Mapping files (`node_token_match.txt`, `kmer_token_id.txt`) linking token IDs to genomic features
  - SHAP-filtered feature lists (`*_shap_filter.txt`)
- **Model training** (`StrainAMR_model_train.py`)
  - Results are grouped into subfolders within the specified `--outdir`
    - `models/` – checkpoints such as `best_model_f1_score.pt`
    - `logs/` – training logs and per-sample probability outputs
    - `shap/` – SHAP interaction pair files (`strains_train_*_interaction.txt`)
    - `analysis/` – attention-weight graphs and top-token tables
- **Prediction** (`StrainAMR_model_predict.py`)
  - Results saved under the specified `--outdir`
    - `logs/` – prediction summaries and per-sample probabilities
    - `shap/` – SHAP value tables and interaction scores for test genomes
    - `analysis/` – attention-weight graphs and top-token tables for predictions

## New Features

- `StrainAMR_build_train.py` accepts `--threads` to process genomes in parallel
- Model training computes SHAP interaction values and maps token IDs back to genomic features for improved interpretability

## Citation

If you use StrainAMR in your research, please cite:

> Liao et al. *StrainAMR: ...* (2024)

