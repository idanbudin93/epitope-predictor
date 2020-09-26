import unittest
from os import path
from typing import List, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from model.Epitope import Epitope
from model.EpitopesDataset import EpitopesDataset


EPITOPES_BATCHES_PATHS = \
    [
        path.abspath('res\\epitopes_batch1.fasta'),
        path.abspath('res\\epitopes_batch2.fasta'),
        path.abspath('res\\epitopes_batch3.fasta')
    ]

WRITE_DATASET_RES_PATH = path.abspath('output\\epitopes_dataset_written.fasta')


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
