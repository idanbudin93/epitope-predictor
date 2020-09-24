import unittest
from os import path

from Bio.SeqRecord import SeqRecord

from models.Epitope import Epitope
from models.EpitopesClusters import EpitopesClusters

EPITOPES_CLUSTERS_FNAME = 'epitopes_clusters1.clstr'
EPITOPES_FASTA_FNAME = 'epitopes1.fasta'

RES_DIR_REL_PATH = 'res'

RES_DIR_PATH = path.abspath(RES_DIR_REL_PATH)

EPITOPES_CLUSTERS_PATH = path.join(RES_DIR_PATH, EPITOPES_CLUSTERS_FNAME)
EPITOPES_FASTA_PATH = path.join(RES_DIR_PATH, EPITOPES_FASTA_FNAME)


class TestInit(unittest.TestCase):
    def test_init(self):
        expected_epitopes_clusters_lst = [
            [Epitope(SeqRecord('AaAA')), Epitope(SeqRecord('aaaa'))],
            [Epitope(SeqRecord('bBBBB'))],
            [Epitope(SeqRecord('ddDDD')), Epitope(SeqRecord('DDdDdD')), Epitope(SeqRecord('DDdddD'))]
        ]

        actual_epitopes_clusters = EpitopesClusters(EPITOPES_CLUSTERS_PATH, EPITOPES_FASTA_PATH)

        self.assertEqual(expected_epitopes_clusters_lst, list(actual_epitopes_clusters))


class TestGetNumOfEpitopes(unittest.TestCase):
    def test_get_num_of_epitopes(self):
        expected_num_of_epitopes = 6

        epitopes_clusters = EpitopesClusters(EPITOPES_CLUSTERS_PATH, EPITOPES_FASTA_PATH)
        actual_num_of_epitopes = epitopes_clusters.get_num_of_epitopes()

        self.assertEqual(expected_num_of_epitopes, actual_num_of_epitopes)


if __name__ == '__main__':
    unittest.main()