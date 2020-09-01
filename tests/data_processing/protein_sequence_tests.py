import unittest

from biotite.sequence.align import SubstitutionMatrix

from model.protein_sequence import ProteinSequence

SUBSTITUTION_MATRIX = SubstitutionMatrix.std_protein_matrix()
NONHOMOLOGOUS_THRESHOLD = 0.8


class TestObjectMethods(unittest.TestCase):
    def test_equal(self):
        protein_seq1 = ProteinSequence('AAABC')
        protein_seq2 = ProteinSequence('AAABC')

        self.assertEqual(protein_seq1, protein_seq2)

    def test_equal_case_sensitive(self):
        protein_seq1 = ProteinSequence('AAABC')
        protein_seq2 = ProteinSequence('AAaBC')

        self.assertNotEqual(protein_seq1, protein_seq2)

    def test_str(self):
        str_seq = 'AAABC'
        protein_seq = ProteinSequence(str_seq)

        self.assertEqual(str(protein_seq), str_seq)

    def test_len(self):
        protein_seq = ProteinSequence('AAABC')

        self.assertEqual(len(protein_seq), 5)


class TestIsHomolog(unittest.TestCase):
    def test_is_homolog1(self):
        protein_seq1 = ProteinSequence('AAABC')
        protein_seq2 = ProteinSequence('AACBC')

        self.assertTrue(protein_seq1.is_homolog(protein_seq2,
                                                substitution_matrix=SUBSTITUTION_MATRIX,
                                                threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_is_homolog2(self):
        protein_seq1 = ProteinSequence('AAAAABC')
        protein_seq2 = ProteinSequence('AAAaC')

        self.assertTrue(protein_seq1.is_homolog(protein_seq2,
                                                substitution_matrix=SUBSTITUTION_MATRIX,
                                                threshold=NONHOMOLOGOUS_THRESHOLD))

    def test_not_is_homolog(self):
        protein_seq1 = ProteinSequence('ABCD')
        protein_seq2 = ProteinSequence('ABCE')

        self.assertFalse(protein_seq1.is_homolog(protein_seq2,
                                                 substitution_matrix=SUBSTITUTION_MATRIX,
                                                 threshold=NONHOMOLOGOUS_THRESHOLD))


if __name__ == '__main__':
    unittest.main()
