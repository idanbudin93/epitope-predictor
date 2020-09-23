__author__ = 'Idan Budin'

import unittest
from os import path

from biotite.sequence.align import SubstitutionMatrix

from data_processing import dataset

CURR_DIR = path.dirname(path.realpath(__file__))
TESTS_DATA_DIR = 'tests_data'
VALID_FASTA_FNAME = 'valid_fata.fasta'
VALID_FASTA_WITH_EMPTY_LINE_FNAME = 'valid_fata_with_empty_line.fasta'
SUBSTITUTION_MATRIX = SubstitutionMatrix.std_protein_matrix()
NONHOMOLOGOUS_THRESHOLD = 0.8
TESTS_OUTPUT_DIR = 'tests_output'
SEQS_DICT_TO_FASTA_RES_FNAME = 'output_fasta.fasta'


class TestInitDataset(unittest.TestCase):
    def test_valid_fasta(self):
        expected = {
            ('SEQ1', 'AcdEFGHIKLMNPQRST'),
            ('SEQ2', 'DEFGGIKLMN'),
            ('SEQ3', 'DEFGIIKLMN'),
            ('SEQ4', 'GGGGGGGGGGGGGGGGGG')
        }

        fasta_path = path.join(CURR_DIR, TESTS_DATA_DIR, VALID_FASTA_FNAME)
        result = set(dataset.Dataset(fasta_path).sequences())

        self.assertSetEqual(expected, result)

    def test_valid_fasta_with_empty_line(self):
        expected = {
            ('SEQ1', 'AcdEFGHIKLMNPQRST'),
            ('SEQ2', 'DEFGGIKLMN'),
            ('SEQ3', 'DEFGIIKLMN'),
            ('SEQ4', 'GGGGGGGGGGGGGGGGGG')
        }

        fasta_path = path.join(CURR_DIR, TESTS_DATA_DIR,
                               VALID_FASTA_FNAME)
        result = set(dataset.Dataset(fasta_path).sequences())

        self.assertSetEqual(expected, result)


class TestRemoveHomologs(unittest.TestCase):
    def test_remove_homologs(self):
        fasta_text = '\n'.join([
            '>SEQ1',
            'AAAA',
            '>SEQ2',
            'AAAAAAAAAA',
            '>SEQ3',
            'AAAA',
            '>SEQ4',
            'ABAABA',
            '>SEQ5',
            'AbACABAAB',
            '>SEQ6',
            'BAAD',
            '>SEQ7',
            'P'
        ])

        expected = {
            ('SEQ2', 'AAAAAAAAAA'),
            ('SEQ5', 'AbACABAAB'),
            ('SEQ6', 'BAAD'),
            ('SEQ7', 'P')
        }

        fasta_dataset = dataset.Dataset(fasta_text)
        fasta_dataset.remove_homologs(substitution_matrix=SUBSTITUTION_MATRIX, threshold=NONHOMOLOGOUS_THRESHOLD)
        result = set(fasta_dataset.sequences())

        self.assertSetEqual(result, expected)


class TestWriteFasta(unittest.TestCase):
    def test_seqs_dict_to_fasta(self):
        fasta_text = '\n'.join([
            '>SEQ1',
            'AAAA',
            '>SEQ2',
            'ABBBBBBBBBBBBBBBBBBBBBBBCCCCCCCCA',
            '>SEQ3 [DESCTIPTION]',
            'p'
        ])

        output_path = path.join(CURR_DIR, TESTS_OUTPUT_DIR,
                                SEQS_DICT_TO_FASTA_RES_FNAME)
        fasta_dataset = dataset.Dataset(fasta_text)
        fasta_dataset.write_fasta(output_path)

        with open(output_path) as res_file:
            res_file_text = res_file.read()

        expected_file_text = '\n'.join([
            ">SEQ1",
            "AAAA",
            ">SEQ2",
            "ABBBBBBBBBBBBBBBBBBBBBBBCCCCCCCCA",
            ">SEQ3 [DESCTIPTION]",
            "p"
        ])

        self.assertEqual(res_file_text, expected_file_text)


if __name__ == '__main__':
    unittest.main()
