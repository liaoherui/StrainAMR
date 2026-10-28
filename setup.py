#!/usr/bin/env python3
"""Install StrainAMR so that `strainamr` is on PATH.

    pip install -e .

The pipeline also calls external tools that pip cannot install: prodigal, rgi (CARD RGI), mmseqs
(or cd-hit), minimap2, seqwish, mash and, for the known-mechanism layer, NCBI AMRFinderPlus
(`conda install -c bioconda ncbi-amrfinderplus && amrfinder -u`). See README.md.
"""

from setuptools import setup

setup(
    name="StrainAMR",
    version="1.0",
    description="Antimicrobial resistance prediction from bacterial genomes",
    url="https://github.com/liaoherui/StrainAMR",
    py_modules=[
        "strainamr",
        "StrainAMR_build_train",
        "StrainAMR_build_test",
        "StrainAMR_model_train",
        "StrainAMR_model_predict",
        "StrainAMR_interpret",
        "kmer_select",
        "pc_clusters",
        "rgi_batch",
        "steplog",
        "feature_selection_sp",
        "feature_selection_sp_test",
        "cal_length_fs",
        "cal_length_test_fs",
        "extract_seq_for_graph",
        "build_graph_batch_minimap2",
        "align_genome_to_graph",
        "generate_token_from_graph",
        "generate_token_from_alignment",
        "generate_token_from_ps",
        "generate_token_from_ps_predict",
        "generate_token_from_kdm",
        "generate_token_from_kdm_predict",
        "sample_ids",
        "mechanism_layer",
        "slim_db",
        "mechanism_eval",
    ],
    packages=["library"],
    install_requires=["numpy", "scipy", "scikit-learn", "pandas", "torch", "biopython", "pyrodigal"],
    entry_points={"console_scripts": ["strainamr = strainamr:main"]},
)
