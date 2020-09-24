import random
import unittest
from os import path
from typing import List, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from data_processing.datasets_processing import (
    count_records_with_adjacent_verified_regions,
    count_total_adjacent_verified_regions,
    count_verified_regions_in_records_with_adjacent_verified_regions, get_epitopes_with_max_verified_regions,
    split_epitopes_clusters_to_cv_datasets
)
from model.Epitope import Epitope
from model.EpitopesClusters import EpitopesClusters
from model.EpitopesDataset import EpitopesDataset


EPITOPES_BATCHES_PATHS = \
    [
        path.abspath('res\\epitopes_batch1.fasta'),
        path.abspath('res\\epitopes_batch2.fasta'),
        path.abspath('res\\epitopes_batch3.fasta')
    ]

EPITOPES_CLUSTERS1_PATH = path.abspath('res\\epitopes_clusters1.clstr')
EPITOPES_CLUSTERS2_PATH = path.abspath('res\\epitopes_clusters2.clstr')
EPITOPES_FASTA1_PATH = path.abspath('res\\epitopes1.fasta')
EPITOPES_FASTA2_PATH = path.abspath('res\\epitopes2.fasta')


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

        epitopes_clusters = EpitopesClusters(EPITOPES_CLUSTERS1_PATH, EPITOPES_FASTA1_PATH)
        actual_epitopes_dataset = get_epitopes_with_max_verified_regions(epitopes_clusters)

        self.assertEqual(expected_epitopes_dataset, actual_epitopes_dataset)


class TestSplitEpitopesClustersToCvGroups(unittest.TestCase):
    def test_split_epitopes_clusters_to_cv_groups_cv10(self):
        cv_fold = 10
        expected_epitopes_cv_datasets = [
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('aaaAA'))),
                    Epitope(SeqRecord(Seq('Aaa'))),
                    Epitope(SeqRecord(Seq('bbbBC'))),
                    Epitope(SeqRecord(Seq('cccaaCd'))),
                    Epitope(SeqRecord(Seq('DcDcDc'))),
                    Epitope(SeqRecord(Seq('AAAAA')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('aaaG'))),
                    Epitope(SeqRecord(Seq('GCAcGcGa'))),
                    Epitope(SeqRecord(Seq('aCGPfpc'))),
                    Epitope(SeqRecord(Seq('cccccCCCccc'))),
                    Epitope(SeqRecord(Seq('GgG'))),
                    Epitope(SeqRecord(Seq('DDDDD')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('EEeeeGGGDDD'))),
                    Epitope(SeqRecord(Seq('BBBbbb'))),
                    Epitope(SeqRecord(Seq('NMnMnM'))),
                    Epitope(SeqRecord(Seq('KPkgK'))),
                    Epitope(SeqRecord(Seq('AAAaaA'))),
                    Epitope(SeqRecord(Seq('AAAaaaA')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('BBBbbBbB'))),
                    Epitope(SeqRecord(Seq('CCCccC'))),
                    Epitope(SeqRecord(Seq('GGGggGG'))),
                    Epitope(SeqRecord(Seq('CcCcCc'))),
                    Epitope(SeqRecord(Seq('cCcccc'))),
                    Epitope(SeqRecord(Seq('ccCccC'))),
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('CccCC'))),
                    Epitope(SeqRecord(Seq('cccCCcC'))),
                    Epitope(SeqRecord(Seq('CccCCCccC'))),
                    Epitope(SeqRecord(Seq('CcccCCcccc'))),
                    Epitope(SeqRecord(Seq('cccCCCccC'))),
                    Epitope(SeqRecord(Seq('cccCccccc')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('aaAAAaaAA'))),
                    Epitope(SeqRecord(Seq('BBBBbbBB'))),
                    Epitope(SeqRecord(Seq('bbbBBBBB'))),
                    Epitope(SeqRecord(Seq('GGGGgg'))),
                    Epitope(SeqRecord(Seq('GGGG')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('TTTtTTT'))),
                    Epitope(SeqRecord(Seq('HHHHHHhhh'))),
                    Epitope(SeqRecord(Seq('HHHhhhhKK'))),
                    Epitope(SeqRecord(Seq('kkkkKKKk'))),
                    Epitope(SeqRecord(Seq('UUUuuuU'))),
                    Epitope(SeqRecord(Seq('GFgGgF'))),
                    Epitope(SeqRecord(Seq('CCCcCBBb'))),
                    Epitope(SeqRecord(Seq('mmmmmmMMMmm'))),
                    Epitope(SeqRecord(Seq('BBbb')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('GGGg'))),
                    Epitope(SeqRecord(Seq('AAaa'))),
                    Epitope(SeqRecord(Seq('AAAa')))
                ])
        ]

        epitopes_clusters = EpitopesClusters(EPITOPES_CLUSTERS2_PATH, EPITOPES_FASTA2_PATH)
        actual_epitopes_cv_datasets = split_epitopes_clusters_to_cv_datasets(
            epitopes_clusters,
            cv_fold,
            shuffle_clusters=False)

        self.assertEqual(expected_epitopes_cv_datasets, actual_epitopes_cv_datasets)

    def test_split_epitopes_clusters_to_cv_groups_cv5(self):
        cv_fold = 5
        expected_epitopes_cv_datasets = [
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('aaaAA'))),
                    Epitope(SeqRecord(Seq('Aaa'))),
                    Epitope(SeqRecord(Seq('bbbBC'))),
                    Epitope(SeqRecord(Seq('cccaaCd'))),
                    Epitope(SeqRecord(Seq('DcDcDc'))),
                    Epitope(SeqRecord(Seq('AAAAA'))),
                    Epitope(SeqRecord(Seq('aaaG'))),
                    Epitope(SeqRecord(Seq('GCAcGcGa'))),
                    Epitope(SeqRecord(Seq('aCGPfpc'))),
                    Epitope(SeqRecord(Seq('cccccCCCccc'))),
                    Epitope(SeqRecord(Seq('GgG'))),
                    Epitope(SeqRecord(Seq('DDDDD'))),
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('EEeeeGGGDDD'))),
                    Epitope(SeqRecord(Seq('BBBbbb'))),
                    Epitope(SeqRecord(Seq('NMnMnM'))),
                    Epitope(SeqRecord(Seq('KPkgK'))),
                    Epitope(SeqRecord(Seq('AAAaaA'))),
                    Epitope(SeqRecord(Seq('AAAaaaA'))),
                    Epitope(SeqRecord(Seq('BBBbbBbB'))),
                    Epitope(SeqRecord(Seq('CCCccC'))),
                    Epitope(SeqRecord(Seq('GGGggGG'))),
                    Epitope(SeqRecord(Seq('CcCcCc'))),
                    Epitope(SeqRecord(Seq('cCcccc'))),
                    Epitope(SeqRecord(Seq('ccCccC'))),
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('CccCC'))),
                    Epitope(SeqRecord(Seq('cccCCcC'))),
                    Epitope(SeqRecord(Seq('CccCCCccC'))),
                    Epitope(SeqRecord(Seq('CcccCCcccc'))),
                    Epitope(SeqRecord(Seq('cccCCCccC'))),
                    Epitope(SeqRecord(Seq('cccCccccc'))),
                    Epitope(SeqRecord(Seq('aaAAAaaAA'))),
                    Epitope(SeqRecord(Seq('BBBBbbBB'))),
                    Epitope(SeqRecord(Seq('bbbBBBBB'))),
                    Epitope(SeqRecord(Seq('GGGGgg')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('GGGG'))),
                    Epitope(SeqRecord(Seq('TTTtTTT'))),
                    Epitope(SeqRecord(Seq('HHHHHHhhh'))),
                    Epitope(SeqRecord(Seq('HHHhhhhKK'))),
                    Epitope(SeqRecord(Seq('kkkkKKKk'))),
                    Epitope(SeqRecord(Seq('UUUuuuU'))),
                    Epitope(SeqRecord(Seq('GFgGgF'))),
                    Epitope(SeqRecord(Seq('CCCcCBBb'))),
                    Epitope(SeqRecord(Seq('mmmmmmMMMmm'))),
                    Epitope(SeqRecord(Seq('BBbb')))
                ]),
            EpitopesDataset(
                [
                    Epitope(SeqRecord(Seq('GGGg'))),
                    Epitope(SeqRecord(Seq('AAaa'))),
                    Epitope(SeqRecord(Seq('AAAa')))
                ])
        ]

        epitopes_clusters = EpitopesClusters(EPITOPES_CLUSTERS2_PATH, EPITOPES_FASTA2_PATH)
        actual_epitopes_cv_datasets = split_epitopes_clusters_to_cv_datasets(
            epitopes_clusters,
            cv_fold,
            shuffle_clusters=False)

        self.assertEqual(expected_epitopes_cv_datasets, actual_epitopes_cv_datasets)


if __name__ == '__main__':
    unittest.main()
