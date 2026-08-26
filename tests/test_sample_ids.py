import os
import tempfile
import unittest

from sample_ids import index_genomes, sample_id_from_filename, validate_label_genomes


class SampleIdTests(unittest.TestCase):
    def test_normalizes_ncbi_assembly_filename(self):
        self.assertEqual(
            sample_id_from_filename("GCF_900125395.1_17175_1_33_genomic.fna"),
            "GCF_900125395",
        )
        self.assertEqual(sample_id_from_filename("GCA_123456.2_genomic.fa"), "GCA_123456")
        self.assertEqual(sample_id_from_filename("custom.sample.fasta"), "custom.sample")

    def test_validates_matching_labels_after_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            genome_dir = os.path.join(directory, "genomes")
            os.mkdir(genome_dir)
            genome = os.path.join(genome_dir, "GCF_900125395.1_name_genomic.fna")
            label = os.path.join(directory, "labels.tsv")
            open(genome, "w").close()
            with open(label, "w") as handle:
                handle.write("ID\tLabel\nGCF_900125395\t1\n")
            genomes = index_genomes(genome_dir)
            self.assertEqual(validate_label_genomes(label, genomes), ["GCF_900125395"])

    def test_reports_missing_and_unlabelled_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            label = os.path.join(directory, "labels.tsv")
            with open(label, "w") as handle:
                handle.write("ID\tLabel\nexpected\t1\n")
            with self.assertRaisesRegex(ValueError, "expected.*actual"):
                validate_label_genomes(label, {"actual": "/genomes/actual.fna"})

    def test_rejects_normalized_accession_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            open(os.path.join(directory, "GCF_1.1_a_genomic.fna"), "w").close()
            open(os.path.join(directory, "GCF_1.2_b_genomic.fna"), "w").close()
            with self.assertRaisesRegex(ValueError, "Multiple genome files"):
                index_genomes(directory)


if __name__ == "__main__":
    unittest.main()
