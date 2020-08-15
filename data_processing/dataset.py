from os import path
from typing import Dict

import biotite
from biotite import sequence as seq
from biotite.sequence import align
from biotite.sequence.io import fasta


FASTA_EXTENSIONS = ['.fasta', '.fna', '.ffn', '.faa', '.frn']


class Dataset:
    """
    A :class:'Dataset' represents protein sequences dataset, including processing methods

    Parameters
        ----------
        fasta_path : str
            The path of FASTA file

    Raises
        -------
        AssertionError
            If input FASTA path "{fasta_path}" does not exists
        AssertionError
            If input file "{fasta_path}" is not FASTA extension'
        AssertionError
            if input FASTA path "{fasta_path}" has odd number of lines
        AssertionError
            If input FASTA path 'Line {lineNumber} in Input FASTA path
            "{fasta_path}" should start with ">"
        biotite.sequence.AlphabetError
            If Symbol {noProteinSymbol} is not in the alphabet'
    """
    def __init__(self, fasta_path: str):
        self.__dataset_dict = self._read_fasta(fasta_path)

    def _read_fasta(self, fasta_path: str) -> Dict[str, str]:
        """
        Reads sequences from FASTA file

        Parameters
        ----------
        fasta_path : str
            The path of FASTA file

        Returns
        -------
        sequences_dict : Dict[str, str]
            Input FASTA path invalid

        Raises
        -------
        AssertionError
            If input FASTA path "{fasta_path}" does not exists
        AssertionError
            If input file "{fasta_path}" is not FASTA extension'
        AssertionError
            if input FASTA path "{fasta_path}" has odd number of lines
        AssertionError
            If input FASTA path 'Line {lineNumber} in Input FASTA path
            "{fasta_path}" should start with ">"
        biotite.sequence.AlphabetError
            If Symbol {noProteinSymbol} is not in the alphabet'
        """

        def parse_key_line(key_line: str) -> str:
            key_line_with_no_prefix = key_line[1:].strip()

            return key_line_with_no_prefix

        assert path.isfile(fasta_path), \
            f'Input FASTA path "{fasta_path}" does not exists'
        assert path.splitext(fasta_path)[1] in FASTA_EXTENSIONS, \
            f'Input file "{fasta_path}" is not FASTA extension'

        sequences_dict = dict()

        with open(fasta_path, 'r') as fasta_file:
            lines = fasta_file.readlines()

            # removing last line if empty
            if lines[-1].strip() == '':
                del lines[-1]

            assert len(lines) % 2 == 0, f'Input FASTA path "{fasta_path}" has ' \
                                        + 'odd number of lines'

            for i in range(int(len(lines) / 2)):
                key_line = lines[i * 2]
                val_line = lines[i * 2 + 1]

                assert key_line.startswith('>'), f'Line {i + 1} in Input FASTA ' \
                                                 + 'path "{fasta_path}" should ' \
                                                 + 'start with ">"'

                key = key_line[1:].strip()
                val = val_line.strip()
                # asserting that value is legal Protein Sequence
                seq.ProteinSequence(val)
                sequences_dict.update({key: val})

        return sequences_dict


    def write_fasta(output_fasta_path: str,
                    seqs_dict: Dict[str, str]):
        """
        Saves a Dictionary of Pritein Sequences as FASTA file

        Parameters
        ----------
        output_fasta_path : str
            The path of the output FASTA file
        seqs_dict: Dict[str, str]
            A Dictionary of protein sequence

        Raises:
        -------
        AssertionError
            If Output Directory
            "{os.path.abspath(path.dirname(output_fasta_path))}" Does not exists'
        AssertionError
            If input file "{output_fasta_path}" is not FASTA extension
        biotite.sequence.AlphabetError
            If Symbol {noProteinSymbol} is not in the alphabet
        """
        assert path.isdir(path.abspath(path.dirname(output_fasta_path))), \
            f'Output Directory \
                "{path.abspath(path.dirname(output_fasta_path))}" \
                Does not exists'
        assert path.splitext(output_fasta_path)[1] in FASTA_EXTENSIONS, \
            f'Input file "{output_fasta_path}" is not FASTA extension'

        lines = []

        for key, val in seqs_dict.items():
            # asserting that value is legal Protein Sequence
            seq.ProteinSequence(val)
            lines.append('>' + key)
            lines.append(val)

        text = '\n'.join(lines)

        with open(output_fasta_path, 'w') as fasta_file:
            fasta_file.writelines(text)


    def get_blosum62_substitution_matrix(self) -> align.SubstitutionMatrix:
        """
        Returns BLOSUM62 proteins substitution matrix

        Returns
        -------
        blosum62_substitution_matrix : biotite.sequence.align.SubstitutionMatrix
            A BLOSUM62 proteins substitution matrix
        """
        blosum62_substitution_matrix = \
            align.SubstitutionMatrix.std_protein_matrix()
        return blosum62_substitution_matrix


    def are_seqs_homologs(seq1: str,
                          seq2: str,
                          substitution_matrix: align.SubstitutionMatrix,
                          threshold: float = 0.8) -> bool:
        """
        Checks whether two protein sequences are homologs

        Parameters
        ----------
        seq1 : str
             A protein sequence
        seq2 : str
            A protein sequence
        substitution_matrix : biotite.sequence.align.SubstitutionMatrix
            A substitution matrix for scoring
        threshold : float, optional
            A treshold of nonhomolgous sequences (Default: 0.8)

        Returns
        -------
        are_homologs : bool
            Whether the two sequences are homologs

        Raises
        -------
        AssertionError
            If homologs threshold {threshold} is not between 0 and 1'
        """
        assert 0 <= threshold <= 1, f'Homologs threshold {threshold} is not \
            between 0 and 1'

        prot_seq1 = seq.ProteinSequence(seq1)
        prot_seq2 = seq.ProteinSequence(seq2)

        alignment = align.align_optimal(prot_seq1, prot_seq2, substitution_matrix,
                                        terminal_penalty=False)[0]

        try:
            identity = align.get_sequence_identity(alignment)
            if identity >= threshold:
                are_homologs = True
            else:
                are_homologs = False
        except ValueError:
            are_homologs = False

        return are_homologs


    def remove_homologs(seqs_dict: Dict[str, str],
                        substitution_matrix: align.SubstitutionMatrix,
                        threshold: float = 0.8):
        """
        Removes homolog sequences from dictionary of ProteinSequence
        (for each pair of homologs the shorter sequence is removed)

        Parameters
        ----------
        seqs_dict : seqs_dict: Dict[str, str]
            A Dictionary of protein sequences
        substitution_matrix : biotite.sequence.align.SubstitutionMatrix
            A substitution matrix for scoring the sequences alignment
        threshold : float, optional
             A treshold of nonhomolgous sequences (Default: 0.8)


        Raises
        ------
        AssertionError
            If homologs threshold {threshold} is not between 0 and 1'
        """
        assert 0 <= threshold <= 1, f'Homologs threshold {threshold} is not \
            between 0 and 1'

        seqs_items_list = list(seqs_dict.items())

        for i in range(1, len(seqs_items_list)):
            for j in range(i):
                key1, seq1 = seqs_items_list[i]
                key2, seq2 = seqs_items_list[j]
                if key1 in seqs_dict and key2 in seqs_dict:
                    if are_seqs_homologs(self, seq1, seq2, substitution_matrix,
                                         threshold=threshold):
                        shorter_seq_list_ind = i if len(seq1) < len(seq2) else j
                        shorter_seq_dict_key = \
                            seqs_items_list[shorter_seq_list_ind][0]
                        del seqs_dict[shorter_seq_dict_key]
