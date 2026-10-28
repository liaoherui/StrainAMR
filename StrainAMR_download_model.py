"""Download one released StrainAMR model bundle from Google Drive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "models" / "manifest.tsv"

SPECIES_ALIASES = {
    "aba": "Acinetobacter baumannii",
    "cdi": "Clostridioides difficile",
    "ecoli": "Escherichia coli",
    "kcp": "Klebsiella pneumoniae",
    "kpn": "Klebsiella pneumoniae",
    "mtb": "Mycobacterium tuberculosis",
    "sau": "Staphylococcus aureus",
}


def normalized(value):
    return "".join(ch for ch in value.lower() if ch.isalnum())


def load_manifest(path=DEFAULT_MANIFEST):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def species_matches(requested, row):
    canonical = SPECIES_ALIASES.get(normalized(requested), requested)
    prefix = row["dataset"].split("_", 1)[0]
    return normalized(canonical) in {
        normalized(row["species"]),
        normalized(prefix),
    }


def select_models(rows, species=None, drug=None, all_models=False):
    if all_models:
        return rows
    if not species or not drug:
        raise ValueError("choose one model with both -s/--species and -d/--drug, or use --all")
    selected = [
        row for row in rows
        if species_matches(species, row) and normalized(drug) == normalized(row["drug"])
    ]
    if not selected:
        available = ", ".join(
            sorted(f'{row["species"]}: {row["drug"]}' for row in rows)
        )
        raise ValueError(f"no released model for species={species!r}, drug={drug!r}; available: {available}")
    return selected


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath([str(destination), str(target)]) != str(destination):
                raise ValueError(f"unsafe path in {archive}: {member.filename}")
        bundle.extractall(destination)


def download_one(row, destination, unzip=True):
    drive_id = row.get("google_drive_id", "").strip()
    if not drive_id:
        raise ValueError(f'no Google Drive ID recorded for {row["file"]}')

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required; install dependencies with: pip install -r requirements.txt") from exc

    archives = destination / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    target = archives / row["file"]
    expected = row.get("sha256", "").strip().lower()

    if target.exists() and expected and sha256(target) == expected:
        print(f"have     {target}")
    else:
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            partial.unlink()
        print(f'download {row["species"]} / {row["drug"]}: {row["file"]}')
        result = gdown.download(id=drive_id, output=str(partial), quiet=False)
        if not result or not partial.exists():
            raise RuntimeError(f'Google Drive download failed for {row["file"]}')
        if expected and sha256(partial) != expected:
            partial.unlink()
            raise RuntimeError(f'SHA256 mismatch for {row["file"]}')
        os.replace(partial, target)

    if unzip:
        dataset = row["dataset"]
        model_name = dataset[:-6] if dataset.endswith("_3fold") else dataset
        extracted = destination / model_name
        marker = extracted / "selected_model.txt"
        if marker.exists():
            print(f"have     {extracted}")
        else:
            extracted.mkdir(parents=True, exist_ok=True)
            print(f"extract  {target} -> {extracted}")
            safe_extract(target, extracted)
        return extracted
    return target


def parser():
    result = argparse.ArgumentParser(
        description="Download released StrainAMR model bundles from Google Drive."
    )
    result.add_argument("-s", "--species", help="species name or alias: Aba, Cdi, Ecoli, Kcp, Mtb, Sau")
    result.add_argument("-d", "--drug", help="antimicrobial name")
    result.add_argument("-o", "--outdir", default="downloaded_models", help="destination directory")
    result.add_argument("--all", action="store_true", help="download all 24 bundles explicitly")
    result.add_argument("--list", action="store_true", help="list available species-drug combinations and exit")
    result.add_argument("--no-unzip", action="store_true", help="keep the verified ZIP without extracting it")
    result.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    rows = load_manifest(args.manifest)
    if args.list:
        for row in rows:
            print(f'{row["species"]}\t{row["drug"]}\t{row["file"]}')
        return 0
    if args.all and (args.species or args.drug):
        parser().error("--all cannot be combined with -s/--species or -d/--drug")
    try:
        selected = select_models(rows, args.species, args.drug, args.all)
        destination = Path(args.outdir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for row in selected:
            download_one(row, destination, unzip=not args.no_unzip)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"done: {len(selected)} model bundle(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
