import unittest
from os import path
from typing import List, Tuple

from data_processing.datasets_processing import (
    count_records_with_adjacent_verified_regions,
    count_total_adjacent_verified_regions,
    count_verified_regions_in_records_with_adjacent_verified_regions
)
from models.Epitope import Epitope
from models.EpitopesDataset import EpitopesDataset

EPITOPES_BATCH1_FNAME = 'epitopes_batch1.fasta'
EPITOPES_BATCH2_FNAME = 'epitopes_batch2.fasta'
EPITOPES_BATCH3_FNAME = 'epitopes_batch3.fasta'
RES_DIR_REL_PATH = 'res'

RES_DIR_PATH = path.abspath(RES_DIR_REL_PATH)

EPITOPES_BATCHES_PATHS = \
    [
        path.join(RES_DIR_PATH, epitope_batch_fname)
        for epitope_batch_fname in [EPITOPES_BATCH1_FNAME, EPITOPES_BATCH2_FNAME, EPITOPES_BATCH3_FNAME]
    ]


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


if __name__ == '__main__':
    unittest.main()
