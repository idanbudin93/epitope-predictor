import unittest
from os import path

from Bio.SeqRecord import SeqRecord

from model.Epitope import Epitope
from model.EpitopesClusters import EpitopesClusters


EPITOPES_CLUSTERS_PATH = path.abspath('res\\epitopes_clusters1.clstr')
EPITOPES_FASTA_PATH = path.abspath('res\\epitopes1.fasta')


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
