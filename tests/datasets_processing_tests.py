import unittest
from os import path
from typing import List, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from data_processing.datasets_processing import (
    count_records_with_adjacent_verified_regions,
    count_total_adjacent_verified_regions,
    count_verified_regions_in_records_with_adjacent_verified_regions, get_epitopes_with_max_verified_regions
)
from models.Epitope import Epitope
from models.EpitopesClusters import EpitopesClusters
from models.EpitopesDataset import EpitopesDataset

EPITOPES_BATCH1_FNAME = 'epitopes_batch1.fasta'
EPITOPES_BATCH2_FNAME = 'epitopes_batch2.fasta'
EPITOPES_BATCH3_FNAME = 'epitopes_batch3.fasta'
EPITOPES_CLUSTERS_FNAME = 'epitopes_clusters.clstr'
EPITOPES_FASTA_FNAME = 'epitopes.fasta'
RES_DIR_REL_PATH = 'res'

RES_DIR_PATH = path.abspath(RES_DIR_REL_PATH)

EPITOPES_BATCHES_PATHS = \
    [
        path.join(RES_DIR_PATH, epitope_batch_fname)
        for epitope_batch_fname in [EPITOPES_BATCH1_FNAME, EPITOPES_BATCH2_FNAME, EPITOPES_BATCH3_FNAME]
    ]

EPITOPES_CLUSTERS_PATH = path.join(RES_DIR_PATH, EPITOPES_CLUSTERS_FNAME)
EPITOPES_FASTA_PATH = path.join(RES_DIR_PATH, EPITOPES_FASTA_FNAME)


def _add_verified_regions_lst(epitope: Epitope, verified_regions_lst: List[Tuple[int, int]]) -> Epitope:
    for verified_region in verified_regions_lst:
        epitope.add_verified_region(verified_region)

    return epitope


class TestCountRecordsWithAdjacentVerifiedRegions(unittest.TestCase):
    def test_count_records_with_adjacent_verified_regions(self):
        expected_records_with_adjacent_verified_regions_count = 4

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.merge_identical_seqs()
        actual_records_with_adjacent_verified_regions_count = \
            count_records_with_adjacent_verified_regions(epitopes_dataset)

        self.assertEqual(expected_records_with_adjacent_verified_regions_count,
                         actual_records_with_adjacent_verified_regions_count)


class TestCountTotalAdjacentVerifiedRegions(unittest.TestCase):
    def test_count_total_adjacent_verified_regions(self):
        expected_total_adjacent_verified_regions_count = 9

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.merge_identical_seqs()
        epitopes_dataset.remove_verified_regions_subsets()
        actual_total_adjacent_verified_regions_count = \
            count_total_adjacent_verified_regions(epitopes_dataset)

        self.assertEqual(expected_total_adjacent_verified_regions_count,
                         actual_total_adjacent_verified_regions_count)


class TestCountVerifiedRegionsInRecordsWithAdjacentVerifiedRegions(unittest.TestCase):
    def test_count_verified_regions_in_records_with_adjacent_verified_regions(self):
        expected_verified_regions_in_records_with_adjacent_verified_regions_count = 9

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.merge_identical_seqs()
        epitopes_dataset.remove_verified_regions_subsets()
        actual_verified_regions_in_records_with_adjacent_verified_regions_count = \
            count_verified_regions_in_records_with_adjacent_verified_regions(epitopes_dataset)

        self.assertEqual(expected_verified_regions_in_records_with_adjacent_verified_regions_count,
                         actual_verified_regions_in_records_with_adjacent_verified_regions_count)


class TestGetEpitopesWithMaxVerifiedRegions(unittest.TestCase):
    def test_get_epitopes_with_max_verified_regions(self):
        expected_epitopes_dataset = EpitopesDataset(
            [
                Epitope(SeqRecord(Seq('AaAA'))),
                Epitope(SeqRecord(Seq('bBBBB'))),
                Epitope(SeqRecord(Seq('DDdDdD')))
            ]
        )

        epitopes_clusters = EpitopesClusters(EPITOPES_CLUSTERS_PATH, EPITOPES_FASTA_PATH)
        actual_epitopes_dataset = get_epitopes_with_max_verified_regions(epitopes_clusters)

        self.assertEqual(expected_epitopes_dataset, actual_epitopes_dataset)


if __name__ == '__main__':
    unittest.main()
