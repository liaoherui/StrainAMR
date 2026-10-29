import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import StrainAMR_download_model as download_model


class DownloadModelTests(unittest.TestCase):
    def test_manifest_has_24_unique_drive_files(self):
        rows = download_model.load_manifest(ROOT / "models" / "manifest.tsv")
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["file"] for row in rows}), 24)
        self.assertEqual(len({row["google_drive_id"] for row in rows}), 24)
        self.assertTrue(all(row["google_drive_id"] for row in rows))

    def test_selects_species_alias_and_drug(self):
        rows = download_model.load_manifest(ROOT / "models" / "manifest.tsv")
        selected = download_model.select_models(rows, "Sau", "erythromycin")
        self.assertEqual([row["file"] for row in selected], ["Sau_erythromycin_3fold__Fold2.zip"])
        selected = download_model.select_models(rows, "Staphylococcus aureus", "ERYTHROMYCIN")
        self.assertEqual([row["file"] for row in selected], ["Sau_erythromycin_3fold__Fold2.zip"])

    def test_requires_explicit_selection_or_all(self):
        rows = download_model.load_manifest(ROOT / "models" / "manifest.tsv")
        with self.assertRaisesRegex(ValueError, "both -s/--species and -d/--drug"):
            download_model.select_models(rows)
        self.assertEqual(len(download_model.select_models(rows, all_models=True)), 24)

    def test_download_verifies_checksum_and_extracts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture = directory / "fixture.zip"
            with zipfile.ZipFile(fixture, "w") as archive:
                archive.writestr("selected_model.txt", "models/model.pt\n")
                archive.writestr("models/model.pt", b"checkpoint")
            expected = hashlib.sha256(fixture.read_bytes()).hexdigest()

            def fake_download(id, output, quiet):
                self.assertEqual(id, "drive-id")
                shutil.copyfile(fixture, output)
                return output

            old_gdown = sys.modules.get("gdown")
            sys.modules["gdown"] = types.SimpleNamespace(download=fake_download)
            try:
                row = {
                    "dataset": "Sau_test_3fold",
                    "species": "Staphylococcus aureus",
                    "drug": "test",
                    "file": "Sau_test.zip",
                    "sha256": expected,
                    "google_drive_id": "drive-id",
                }
                result = download_model.download_one(row, directory / "out")
            finally:
                if old_gdown is None:
                    sys.modules.pop("gdown", None)
                else:
                    sys.modules["gdown"] = old_gdown

            self.assertEqual(result.name, "Sau_test")
            self.assertEqual((result / "selected_model.txt").read_text(), "models/model.pt\n")
            self.assertTrue((directory / "out" / "archives" / "Sau_test.zip").is_file())


if __name__ == "__main__":
    unittest.main()
