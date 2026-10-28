#!/usr/bin/env bash
# Download released StrainAMR model bundles and verify their checksums.
#
#   bash scripts/download_models.sh [dest_dir] [dataset ...]
#
# With no dataset names, every bundle listed in models/manifest.tsv is fetched (about 9 GB).
# Requires curl, unzip and sha256sum.
set -euo pipefail

REPO="liaoherui/StrainAMR"
TAG="v1.0-models"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$here/../models/manifest.tsv"
dest="${1:-models}"
[ $# -gt 0 ] && shift
want=("$@")

[ -f "$manifest" ] || { echo "manifest not found: $manifest" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
mkdir -p "$dest"

selected=0
failed=0

while IFS=$'\t' read -r dataset species drug fold n_val auc f1 tau mechanism file sha; do
  [ "$dataset" = "dataset" ] && continue
  [ -n "${file:-}" ] || continue
  if [ ${#want[@]} -gt 0 ]; then
    keep=0
    for w in "${want[@]}"; do
      [ "$dataset" = "$w" ] && keep=1
    done
    [ "$keep" -eq 1 ] || continue
  fi

  selected=$((selected + 1))
  target="$dest/$file"

  if [ -f "$target" ] && [ -n "${sha:-}" ]; then
    if [ "$(sha256sum "$target" | cut -d' ' -f1)" = "$sha" ]; then
      echo "have    $file"
      continue
    fi
  fi

  url="https://github.com/$REPO/releases/download/$TAG/$file"
  echo "fetch   $file"
  if ! curl -fL --retry 3 --retry-delay 5 -C - -o "$target" "$url"; then
    echo "  download failed: $url" >&2
    failed=$((failed + 1))
    continue
  fi

  if [ -n "${sha:-}" ]; then
    if [ "$(sha256sum "$target" | cut -d' ' -f1)" != "$sha" ]; then
      echo "  checksum mismatch, deleting $target" >&2
      rm -f "$target"
      failed=$((failed + 1))
      continue
    fi
  else
    echo "  warning: no sha256 for $file in the manifest, skipped verification" >&2
  fi

  if command -v unzip >/dev/null; then
    unzip -q -o "$target" -d "$dest/${file%.zip}"
  else
    echo "  unzip not found; the archive is left in $dest" >&2
  fi
done < "$manifest"

echo "done: $selected bundle(s) requested, $failed failure(s)"
[ "$failed" -eq 0 ]
