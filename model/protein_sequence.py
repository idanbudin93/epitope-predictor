from __future__ import annotations

from biotite import sequence as seq
from biotite.sequence import align


class ProteinSequence:
    """
     A :class:`ProteinSequence` represents a protein sequences

    Parameters
    ----------
    str_seq : :class:`str`
        The protein sequences in string representation

    Raises
    -------
    biotite.sequence.AlphabetError
        If Symbol {noProteinSymbol} is not in the alphabet
    """
    def __init__(self, str_seq: str):
        self.__protein_seq = seq.ProteinSequence(str_seq)
        self.__str_seq = str_seq

    def is_homolog(self,
                   other: ProteinSequence,
                   substitution_matrix: align.SubstitutionMatrix,
                   threshold: float = 0.8) -> bool:
        """
        Determines whether other :class:`ProteinSequence` is an homolog sequence

        Parameters
        ----------
        other: :class:`ProteinSequence`
            Another :class:`ProteinSequence`
        substitution_matrix : :class:`align.SubstitutionMatrix`, optional
            A substitution matrix for scoring the sequences alignment
            (Default: the default :class:`align.SubstitutionMatrix` for protein sequence alignments, which is BLOSUM62)
        threshold : :class:`float`, optional
             A threshold of non-homologous sequences (Default: 0.8)

        Returns
        -------
        is_homolog : :class:`bool`
            Whether the other :class:`ProteinSequence` is an homolog sequence

        Raises
        ------
        AssertionError
            If homologs threshold {threshold} is not between 0 and 1'
        """
        assert 0 <= threshold <= 1, f'Homologs threshold {threshold} is not between 0 and 1'

        alignment = align.align_optimal(self.__protein_seq, other.__protein_seq, substitution_matrix, terminal_penalty=False)[0]

        try:
            identity = align.get_sequence_identity(alignment)
            if identity >= threshold:
                is_homolog = True
            else:
                is_homolog = False
        except ValueError:
            is_homolog = False

        return is_homolog

    def __str__(self) -> str:
        return str(self.__str_seq)

    def __len__(self) -> int:
        return len(self.__str_seq)

    def __eq__(self, other: ProteinSequence) -> bool:
        return self.__str_seq == other.__str_seq

    def __hash__(self) -> int:
        return hash(str(self.__str_seq))
