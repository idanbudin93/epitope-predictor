import unittest
from preprocessing.utils import *
from os import path
from typing import Dict
from biotite.sequence import ProteinSequence

CURR_DIR = path.dirname(path.realpath(__file__))
TESTS_DATA_DIR = 'tests_data'
VALID_FASTA_FNAME = 'valid_fata.fasta'
VALID_FASTA_WITH_EMPTY_LINE_FNAME = 'valid_fata_with_empty_line.fasta'
SUBTITUTION_MATRIX = get_blosum62_substitution_matrix()
NONHOMOLOGOUS_THRESHOLD = 0.8
TESTS_OUTPUT_DIR = 'tests_output'
SEQS_DICT_TO_FASTA_RES_FNAME = 'output_fasta.fasta'


class TestReadFasta(unittest.TestCase):
    def test_valid_fasta(self):
        expected = {
            'SEQ1': 'AcdEFGHIKLMNPQRST',
            'SEQ2': 'DEFGGIKLMN',
            'SEQ3': 'DEFGIIKLMN',
            'SEQ4': 'GGGGGGGGGGGGGGGGGG'
        }

        fasta_path = path.join(CURR_DIR, TESTS_DATA_DIR,
                               VALID_FASTA_FNAME)
        result = read_fasta(fasta_path)

        self.assertDictEqual(expected, result)

    def test_valid_fasta_with_empty_line(self):
        expected = {
            'SEQ1': 'AcdEFGHIKLMNPQRST',
            'SEQ2': 'DEFGGIKLMN',
            'SEQ3': 'DEFGIIKLMN',
            'SEQ4': 'GGGGGGGGGGGGGGGGGG'
        }

        fasta_path = path.join(CURR_DIR, TESTS_DATA_DIR,
                               VALID_FASTA_FNAME)
        result = read_fasta(fasta_path)

        self.assertDictEqual(expected, result)


class TestAreSeqsHomologs(unittest.TestCase):
    def test_homologs(self):
        seq1 = 'AAAAA'
        seq2 = 'AAAAB'

        self.assertTrue(are_seqs_homologs(seq1, seq2,
                        SUBTITUTION_MATRIX,
                        threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_homolog_of_subsequence(self):
        seq1 = 'AAPBB'
        seq2 = 'CCCAAGBB'

        self.assertTrue(are_seqs_homologs(seq1, seq2,
                        SUBTITUTION_MATRIX,
                        threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_nonhomolougs1(self):
        seq1 = 'AAAA'
        seq2 = 'BBBBCB'

        self.assertFalse(are_seqs_homologs(seq1, seq2,
                         SUBTITUTION_MATRIX,
                         threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_nonhomolougs2(self):
        seq1 = 'BAAD'
        seq2 = 'AAAAAAAAAA'

        self.assertFalse(are_seqs_homologs(seq1, seq2,
                         SUBTITUTION_MATRIX,
                         threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_empty_seqs(self):
        seq1 = ''
        seq2 = ''

        self.assertFalse(are_seqs_homologs(seq1, seq2,
                         SUBTITUTION_MATRIX,
                         threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_case_insensitive_homologs(self):
        seq1 = 'aaaa'
        seq2 = 'AAAA'

        self.assertTrue(are_seqs_homologs(seq1, seq2,
                        SUBTITUTION_MATRIX, threshold=NONHOMOLOGOUS_THRESHOLD))


class TestRemoveHomologs(unittest.TestCase):
    def test_remove_homologs(self):
        seqs_dict = {
            'SEQ1': 'AAAA',
            'SEQ2': 'AAAAAAAAAA',
            'SEQ3': 'AAAA',
            'SEQ4': 'ABAABA',
            'SEQ5': 'AbACABAAB',
            'SEQ6': 'BAAD',
            'SEQ7': 'P'
        }

        expected = {
            'SEQ2': 'AAAAAAAAAA',
            'SEQ5': 'AbACABAAB',
            'SEQ6': 'BAAD',
            'SEQ7': 'P'
        }

        remove_homologs(seqs_dict, SUBTITUTION_MATRIX,
                        threshold=NONHOMOLOGOUS_THRESHOLD)

        self.assertDictEqual(seqs_dict, expected)


class TestWriteFasta(unittest.TestCase):
    def test_seqs_dict_to_fasta(self):
        seqs_dict = {
            'SEQ1': 'AAAA',
            'SEQ2': 'ABBBBBBBBBBBBBBBBBBBBBBBCCCCCCCCA',
            'SEQ3 [DESCTIPTION]': 'p',
        }

        output_path = path.join(CURR_DIR, TESTS_OUTPUT_DIR,
                                SEQS_DICT_TO_FASTA_RES_FNAME)
        write_fasta(output_path, seqs_dict)
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
