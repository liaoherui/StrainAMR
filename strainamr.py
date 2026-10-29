#!/usr/bin/env python3
"""StrainAMR - antimicrobial resistance prediction from bacterial genomes.

One entry point for the whole pipeline; each subcommand takes the same options the individual
scripts took, so existing command lines keep working with `strainamr <subcommand>` in front.

Typical run:

    strainamr build-train -i genomes_train/ -l labels.txt -d imipenem -o db/ -t 32
    strainamr train       -i db/ -o model/ -t split
    strainamr build-test  -i genomes_test/ -l labels_test.txt -d imipenem -o test/ --db db/ -t 32
    strainamr predict     -i test/ -m model/ -o pred/

`strainamr <subcommand> --help` lists that step's options.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

COMMANDS = {
    "build-train": ("StrainAMR_build_train",
                    "extract graph, PC and k-mer features from the training genomes"),
    "train": ("StrainAMR_model_train",
              "train the multimodal Transformer on the extracted training features"),
    "build-test": ("StrainAMR_build_test",
                   "extract the same features from new genomes, against a built training database"),
    "predict": ("StrainAMR_model_predict",
                "predict resistance for new genomes with a trained model"),
    "interpret": ("StrainAMR_interpret",
                  "attribution and interaction analysis of a trained model"),
    "download-model": ("StrainAMR_download_model",
                       "download one released species-drug model bundle"),
    "check-rgi": ("rgi_batch",
                  "verify that batched RGI reproduces one-run-per-genome RGI on a few genomes"),
    "mechanism-calibrate": ("mechanism_layer",
                            "(re)calibrate the known-mechanism layer of a training database"),
    "mechanism-eval": ("mechanism_eval",
                       "add the known-mechanism layer to existing predictions (no retraining)"),
    "slim-db": ("slim_db",
                "shrink a training database to the files later steps read (dry run by default)"),
}


def usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = [__doc__.strip(), "", "Subcommands:"]
    for name, (_module, help_text) in COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {help_text}")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-v", "--version"):
        print("StrainAMR 1.0")
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        close = [c for c in COMMANDS if c.startswith(cmd[:3])]
        sys.stderr.write(f"unknown subcommand '{cmd}'"
                         + (f"; did you mean {' or '.join(close)}?" if close else "")
                         + "\n\n" + usage() + "\n")
        return 2
    module_name = COMMANDS[cmd][0]
    module = __import__(module_name)
    if cmd == "check-rgi" and not any(x.startswith("--check") for x in argv[1:]):
        argv = argv + ["--check", "3"]          # this subcommand exists only to compare
    # each step still parses sys.argv itself, so hand it the remaining arguments under its own name
    sys.argv = [f"strainamr {cmd}"] + argv[1:]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
