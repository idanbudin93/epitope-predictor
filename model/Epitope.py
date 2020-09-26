from typing import Tuple, List

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class Epitope:
    """
    Epitope with the verified regions in upper case and in a list property

    Parameters
    ----------
    seq_record : Bio.SeqRecord.SeqRecord
        Sequence record with the verified regions in upper case

    Attributes
    ----------
    Epitope.verified_regions
    Epitope.record
    """
    def __init__(self, seq_record: SeqRecord):
        self.__record = seq_record

    def __eq__(self, other):
        return str(self.__record.seq) == str(other.__record.seq)

    def __str__(self):
        return str(self.__record.seq)

    def __hash__(self):
        return str(self.__record.seq).__hash__()

    @property
    def verified_regions(self):
        """
        Returns
        -------
        verified_regions : List[Tuple[int, int]]
            List of the record's verified regions start-end pairs
        """
        verified_regions = []
        seq = self.__get_seq()

        start = 0
        while start < len(seq):
            while start < len(seq) and not seq[start].isupper():
                start += 1

            if start >= len(seq):
                break

            end = start
            while end < len(seq) and seq[end].isupper():
                end += 1

            if end < len(seq) and seq[end].isupper():
                verified_regions.append((start, end))
            else:
                verified_regions.append((start, end - 1))

            start = end + 1

        return verified_regions

    @property
    def record(self) -> SeqRecord:
        """
        Returns
        -------
        record : Bio.SeqRecord.SeqRecord
            Sequence record with the verified regions in upper case
        """
        return self.__record

    def __get_seq(self):
        return str(self.__record.seq)


    def add_verified_region(self, verified_region: Tuple[int, int]):
        """
        Adding verified regions to the record
        Parameters
        ----------
        verified_region : Tuple[int, int]
            Verified regions start-end pairs to add to the record's list
        """

        def mark_verified_region(seq_str: str) -> str:
            start, end = verified_region
            return seq_str[:start] + seq_str[start:end + 1].upper() + seq_str[end + 1:]

        self.verified_regions.append(verified_region)
        self.__record.seq = Seq(mark_verified_region(self.__get_seq()))
