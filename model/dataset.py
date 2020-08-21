__author__ = 'Idan Budin'

from os import path
from typing import Dict, ItemsView

from biotite.sequence import align

from model.protein_sequence import ProteinSequence


class Dataset:
    """
    A :class:`Dataset` represents protein sequences dataset, including processing methods

    Parameters
    ----------
    fasta : :class:`str`
        The path of FASTA file or a FASTS file content

    Raises
    -------
    AssertionError
        If input FASTA path "{fasta_path}" does not exists
    AssertionError
        If input file "{fasta_path}" is not FASTA extension'
    AssertionError
        if input FASTA path "{fasta_path}" has odd number of lines
    AssertionError
        If input FASTA path 'Line {lineNumber} in Input FASTA path "{fasta_path}" should start with ">"
    biotite.sequence.AlphabetError
        If Symbol {noProteinSymbol} is not in the alphabet'
    """

    fasta_extensions = ['.fasta', '.fna', '.ffn', '.faa', '.frn']

    def __init__(self, fasta: str):
        if path.isfile(fasta):
            self.__dataset_dict = self.__read_fasta(fasta)
        else:
            self.__dataset_dict = self.__read_str(fasta)

    def __read_fasta(self, fasta_path: str) -> Dict[str, ProteinSequence]:
        assert path.isfile(fasta_path), f'Input FASTA path "{fasta_path}" does not exists'
        assert path.splitext(fasta_path)[1] in self.fasta_extensions, \
            f'Input file "{fasta_path}" is not FASTA extension'

        with open(fasta_path, 'r') as fasta_file:
            fasta_text = fasta_file.read()
            sequences_dict = self.__read_str(fasta_text)

        return sequences_dict

    def __read_str(self, fasta_text: str) -> Dict[str, ProteinSequence]:
        sequences_dict = dict()

        lines = fasta_text.splitlines()

        # removing last line if empty
        if lines[-1].strip() == '':
            del lines[-1]

        assert len(lines) % 2 == 0, f'Input FASTA has odd number of lines'

        for i in range(int(len(lines) / 2)):
            key_line = lines[i * 2]
            val_line = lines[i * 2 + 1]

            assert key_line.startswith('>'), f'Line {i + 1} in Input FASTA should start with ">"'

            key = key_line[1:].strip()
            val = ProteinSequence(val_line.strip())
            sequences_dict.update({key: val})

        return sequences_dict

    def write_fasta(self, output_fasta_path: str):
        """
        Saves a the dataset as FASTA file

        Parameters
        ----------
        output_fasta_path : :class:`str`
            The path of the output FASTA file

        Raises
        -------
        AssertionError
            If Output Directory "{os.path.abspath(path.dirname(output_fasta_path))}" Does not exists'
        AssertionError
            If input file "{output_fasta_path}" is not FASTA extension
        biotite.sequence.AlphabetError
            If Symbol {noProteinSymbol} is not in the alphabet
        """
        assert path.isdir(path.abspath(path.dirname(output_fasta_path))), \
            f'Output Directory "{path.abspath(path.dirname(output_fasta_path))}" Does not exists'
        assert path.splitext(output_fasta_path)[1] in self.fasta_extensions, \
            f'Input file "{output_fasta_path}" is not FASTA extension'

        text = self.__repr__()

        with open(output_fasta_path, 'w') as fasta_file:
            fasta_file.writelines(text)

    def remove_homologs(self,
                        substitution_matrix: align.SubstitutionMatrix = align.SubstitutionMatrix.std_protein_matrix(),
                        threshold: float = 0.8):
        """
        Removes homologous sequences from the dataset (for each pair of homologs the shorter sequence is removed)

        Parameters
        ----------
        substitution_matrix : :class:`align.SubstitutionMatrix`, optional
            A substitution matrix for scoring the sequences alignment
            (Default: the default :class:`align.SubstitutionMatrix` for protein sequence alignments, which is BLOSUM62)
        threshold : :class:`float`, optional
             A threshold of non-homologous sequences (Default: 0.8)

        Raises
        ------
        AssertionError
            If homologs threshold {threshold} is not between 0 and 1'
        """
        assert 0 <= threshold <= 1, f'Homologs threshold {threshold} is not \
            between 0 and 1'

        seqs_items_list = list(self.sequences())

        for i in range(1, len(seqs_items_list)):
            for j in range(i):
                key1, seq1 = seqs_items_list[i]
                key2, seq2 = seqs_items_list[j]
                if key1 in self.__dataset_dict and key2 in self.__dataset_dict:
                    if seq1.is_homolog(seq2, substitution_matrix, threshold=threshold):
                        shorter_seq_list_ind = i if len(seq1) < len(seq2) else j
                        shorter_seq_dict_key = seqs_items_list[shorter_seq_list_ind][0]
                        del self.__dataset_dict[shorter_seq_dict_key]

    def __repr__(self) -> str:
        lines = []

        for key, val in self.sequences():
            lines.append('>' + key)
            lines.append(str(val))

        text = '\n'.join(lines)

        return text

    def sequences(self) -> ItemsView[str, ProteinSequence]:
        """
        Gets a view object that displays a list of dataset's (sequence_key, sequence) tuple pairs.

        Returns
        ------
        sequences : :class:`ItemsView`[:class:`str`, :class:`str`]
            A view object that displays a list of dataset's (sequence_key, sequence) tuple pairs

        """
        return self.__dataset_dict.items()

    def __len__(self) -> int:
        return len(self.__dataset_dict)
