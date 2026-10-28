#!/usr/bin/env bash
# Backward-compatible entry point for the model downloader.
# Examples:
#   bash scripts/download_models.sh -s Sau -d erythromycin -o pretrained_models/
#   bash scripts/download_models.sh --all -o pretrained_models/
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$here/strainamr.py" download-model "$@"
