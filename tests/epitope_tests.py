import unittest
from typing import List, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from models.Epitope import Epitope


def _add_verified_regions_lst(epitope: Epitope, verified_regions_lst: List[Tuple[int, int]]) -> Epitope:
    for verified_region in verified_regions_lst:
        epitope.add_verified_region(verified_region)

    return epitope


class TestEpitopeInit(unittest.TestCase):
    def test_init1(self):
        seq_example = 'aaaA'

        expected_seq = seq_example
        expected_verified_regions = [(3, 3)]

        actual_epitope = Epitope(SeqRecord(Seq(seq_example)))
        actual_verified_regions = actual_epitope.verified_regions

        self.assertEqual(expected_seq, str(actual_epitope))
        self.assertEqual(expected_verified_regions, actual_verified_regions)

    def test_init2(self):
        seq_example = 'AAaA'

        expected_verified_regions = [(0, 1), (3, 3)]

        epitope = Epitope(SeqRecord(Seq(seq_example)))
        actual_verified_regions = epitope.verified_regions

        self.assertEqual(expected_verified_regions, actual_verified_regions)

    def test_epitope_init_verified_region2(self):
        seq_example = 'AAAA'

        expected_verified_regions = [(0, 3)]

        epitope = Epitope(SeqRecord(Seq(seq_example)))
        actual_verified_regions = epitope.verified_regions

        self.assertEqual(expected_verified_regions, actual_verified_regions)

    def test_epitope_init_verified_region3(self):
        seq_example = 'AAaa'

        expected_verified_regions = [(0, 1)]

        epitope = Epitope(SeqRecord(Seq(seq_example)))
        actual_verified_regions = epitope.verified_regions

        self.assertEqual(expected_verified_regions, actual_verified_regions)


class TestEpitopeEquality(unittest.TestCase):
    def test_epitope_equal(self):
        seq1_example = 'aaaA'
        seq2_example = 'aaaA'

        epitope1 = Epitope(SeqRecord(Seq(seq1_example)))
        epitope2 = Epitope(SeqRecord(Seq(seq2_example)))

        self.assertEqual(epitope1, epitope2)

    def test_epitope_unequal(self):
        seq1_example = 'aaAA'
        seq2_example = 'aaaA'

        epitope1 = Epitope(SeqRecord(Seq(seq1_example)))
        epitope2 = Epitope(SeqRecord(Seq(seq2_example)))

        self.assertNotEqual(epitope1, epitope2)


class TestAddVerifiedRegion(unittest.TestCase):
    def test_add_verified_region(self):
        seq_example = 'AAaa'
        verified_region_to_add_example = (3, 3)

        expected_verified_regions = [(0, 1), (3, 3)]
        expected_record_seq = 'AAaA'

        epitope = Epitope(SeqRecord(Seq(seq_example)))
        epitope.add_verified_region(verified_region_to_add_example)
        actual_verified_regions = epitope.verified_regions
        actual_record_seq = str(epitope.record.seq)

        self.assertEqual(expected_verified_regions, actual_verified_regions)
        self.assertEqual(expected_record_seq, actual_record_seq)


class TestRemoveVerifiedRegionSubsets(unittest.TestCase):
    def test_remove_verified_regions_subsets1(self):
        seq_example = 'AAaa'
        verified_regions_to_add_example = [(0, 1), (0, 0), (1, 1), (1, 2)]

        expected_verified_regions = [(0, 1), (1, 2)]

        epitope = Epitope(SeqRecord(Seq(seq_example)))
        _add_verified_regions_lst(epitope, verified_regions_to_add_example)
        epitope.remove_verified_regions_subsets()
        actual_verified_regions = epitope.verified_regions

        self.assertEqual(expected_verified_regions, actual_verified_regions)

    def test_remove_verified_regions_subsets2(self):
        seq_example = 'aAAa'
        verified_regions_to_add_example = [(1, 1), (0, 0), (0, 1), (0, 0)]

        expected_verified_regions = [(1, 2), (0, 1)]

        epitope = Epitope(SeqRecord(Seq(seq_example)))
        _add_verified_regions_lst(epitope, verified_regions_to_add_example)
        epitope.remove_verified_regions_subsets()
        actual_verified_regions = epitope.verified_regions

        self.assertEqual(expected_verified_regions, actual_verified_regions)


if __name__ == '__main__':
    unittest.main()
