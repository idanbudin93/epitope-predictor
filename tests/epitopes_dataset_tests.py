import unittest
from os import path
from typing import List, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from model.Epitope import Epitope
from model.EpitopesDataset import EpitopesDataset

EPITOPES_BATCH1_FNAME = 'epitopes_batch1.fasta'
EPITOPES_BATCH2_FNAME = 'epitopes_batch2.fasta'
EPITOPES_BATCH3_FNAME = 'epitopes_batch3.fasta'
WRITE_DATASET_RES_FNAME = 'epitopes_dataset_written.fasta'
RES_DIR_REL_PATH = 'res'
OUTPUT_DIR_REL_PATH = 'output'

RES_DIR_PATH = path.abspath(RES_DIR_REL_PATH)
OUTPUT_DIR_PATH = path.abspath(OUTPUT_DIR_REL_PATH)

EPITOPES_BATCHES_PATHS = \
    [
        path.join(RES_DIR_PATH, epitope_batch_fname)
        for epitope_batch_fname in [EPITOPES_BATCH1_FNAME, EPITOPES_BATCH2_FNAME, EPITOPES_BATCH3_FNAME]
    ]

WRITE_DATASET_RES_PATH = path.join(OUTPUT_DIR_PATH, WRITE_DATASET_RES_FNAME)


def add_verified_regions_lst(epitope: Epitope, verified_regions_lst: List[Tuple[int, int]]) -> Epitope:
    for verified_region in verified_regions_lst:
        epitope.add_verified_region(verified_region)

    return epitope


class TestInitEpitopesDataset(unittest.TestCase):
    def test_init_epitopes_dataset(self):
        expected_epitopes = \
            [
                Epitope(SeqRecord(Seq('aaaA'))),
                Epitope(SeqRecord(Seq('aaAA'))),
                Epitope(SeqRecord(Seq('B'))),
                Epitope(SeqRecord(Seq('aaaA'))),
                Epitope(SeqRecord(Seq('aAAa'))),
                Epitope(SeqRecord(Seq('bBBbb'))),
                Epitope(SeqRecord(Seq('bbbBB'))),
                Epitope(SeqRecord(Seq('bbBBb'))),
                Epitope(SeqRecord(Seq('cccCC'))),
                Epitope(SeqRecord(Seq('ccCCc'))),
                Epitope(SeqRecord(Seq('Dd'))),
                Epitope(SeqRecord(Seq('dD')))
            ]

        actual_epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)

        self.assertEqual(expected_epitopes, list(actual_epitopes_dataset))


class TestEquality(unittest.TestCase):
    def test_equal(self):
        epitopes_dataset1 = EpitopesDataset(
            [
                Epitope(SeqRecord(Seq('a'))),
                Epitope(SeqRecord(Seq('A'))),
                Epitope(SeqRecord(Seq('aa'))),
                Epitope(SeqRecord(Seq('aa'))),
            ]
        )

        epitopes_dataset2 = EpitopesDataset(
            [
                Epitope(SeqRecord(Seq('aa'))),
                Epitope(SeqRecord(Seq('A'))),
                Epitope(SeqRecord(Seq('a'))),
            ]
        )

        self.assertEqual(epitopes_dataset1, epitopes_dataset2)

    def test_unequal(self):
        epitopes_dataset1 = EpitopesDataset(
            [
                Epitope(SeqRecord(Seq('a'))),
                Epitope(SeqRecord(Seq('A'))),
                Epitope(SeqRecord(Seq('aa'))),
                Epitope(SeqRecord(Seq('aa'))),
            ]
        )

        epitopes_dataset2 = EpitopesDataset(
            [
                Epitope(SeqRecord(Seq('a'))),
                Epitope(SeqRecord(Seq('B'))),
                Epitope(SeqRecord(Seq('aa'))),
                Epitope(SeqRecord(Seq('aa'))),
            ]
        )

        self.assertNotEqual(epitopes_dataset1, epitopes_dataset2)

class TestMergeIdenticalSeqs(unittest.TestCase):
    def test_merge_identical_seqs(self):
        expected_epitopes = \
            [
                add_verified_regions_lst(Epitope(SeqRecord(Seq('aaaA'))), [(2, 3), (3, 3), (1, 2)]),
                Epitope(SeqRecord(Seq('B'))),
                add_verified_regions_lst(Epitope(SeqRecord(Seq('bBBbb'))), [(3, 4), (2, 3)]),
                add_verified_regions_lst(Epitope(SeqRecord(Seq('cccCC'))), [(2, 3)]),
                add_verified_regions_lst(Epitope(SeqRecord(Seq('Dd'))), [(1, 1)])
            ]

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.merge_identical_seqs()
        actual_epitopes = list(epitopes_dataset)

        self.assertEqual(expected_epitopes, actual_epitopes)


class TestCountVerifiedRegions(unittest.TestCase):
    def test_count_verified_regions(self):
        expected_verified_regions_count = 12

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        actual_verified_regions_count = epitopes_dataset.count_verified_regions()

        self.assertEqual(expected_verified_regions_count, actual_verified_regions_count)


class TestRemoveVerifiedRegionSubsets(unittest.TestCase):
    def test_remove_verified_regions_subsets(self):
        expected_epitopes_verified_regions_lst = \
            [
                ('aAAA', [(2, 3), (1, 2)]),
                ('B', [(0, 0)]),
                ('bBBBB', [(1, 2), (3, 4), (2, 3)]),
                ('ccCCC', [(3, 4), (2, 3)]),
                ('DD', [(0, 0), (1, 1)])
            ]

        expected_epitopes_dataset_len = len(expected_epitopes_verified_regions_lst)

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.merge_identical_seqs()
        epitopes_dataset.remove_verified_regions_subsets()
        actual_epitopes_dataset_len = len(epitopes_dataset)

        self.assertEqual(expected_epitopes_dataset_len, actual_epitopes_dataset_len)
        for i in range(len(expected_epitopes_verified_regions_lst)):
            expected_epitope_seq = expected_epitopes_verified_regions_lst[i][0]
            expected_epitopes_verified_regions = expected_epitopes_verified_regions_lst[i][1]

            actual_epitope_seq = str(epitopes_dataset[i])
            actual_epitopes_verified_regions = epitopes_dataset[i].verified_regions

            self.assertEqual(expected_epitope_seq, actual_epitope_seq)
            self.assertEqual(expected_epitopes_verified_regions, actual_epitopes_verified_regions)


class TestWrite(unittest.TestCase):
    def test_write1(self):
        expected_output_file_text = \
            """>seq1
aaaA
>seq2
aaAA
>seq3
B
>seq1
aaaA
>seq4
aAAa
>seq5
bBBbb
>seq6
bbbBB
>seq7
bbBBb
>seq8
cccCC
>seq9
ccCCc
>seq10
Dd
>seq11
dD
"""

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.write(WRITE_DATASET_RES_PATH)
        with open(WRITE_DATASET_RES_PATH) as output_file:
            actual_output_file_text = output_file.read()

        self.assertEqual(expected_output_file_text,
                         actual_output_file_text)

    def test_write2(self):
        expected_output_file_text = \
            """>seq1
aAAA
>seq3
B
>seq5
bBBBB
>seq8
ccCCC
>seq10
DD
"""

        epitopes_dataset = EpitopesDataset(EPITOPES_BATCHES_PATHS)
        epitopes_dataset.merge_identical_seqs()
        epitopes_dataset.write(WRITE_DATASET_RES_PATH)
        with open(WRITE_DATASET_RES_PATH) as output_file:
            actual_output_file_text = output_file.read()

        self.assertEqual(expected_output_file_text,
                         actual_output_file_text)


if __name__ == '__main__':
    unittest.main()
