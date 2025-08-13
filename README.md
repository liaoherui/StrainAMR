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

## New Features

- `StrainAMR_build_train.py` accepts `--threads` to process genomes in parallel
- Model training computes SHAP interaction values and maps token IDs back to genomic features for improved interpretability

## Citation

If you use StrainAMR in your research, please cite:

> Liao et al. *StrainAMR: ...* (2024)

