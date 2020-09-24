from typing import Tuple, List

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class Epitope:
    def __init__(self, seq_record: SeqRecord):
        self.__record = seq_record
        self.__verified_regions = self.__get_verified_regions()

    def __eq__(self, other):
        return str(self.__record.seq) == str(other.__record.seq)

    def __str__(self):
        return str(self.__record.seq)

    def __hash__(self):
        return str(self.__record.seq).__hash__()

    def __get_seq(self):
        return str(self.__record.seq)

    def __get_verified_regions(self) -> List[Tuple[int, int]]:
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

    def add_verified_region(self, verified_region: Tuple[int, int]):

        def mark_verified_region(seq_str: str) -> str:
            start, end = verified_region
            return seq_str[:start] + seq_str[start:end + 1].upper() + seq_str[end + 1:]

        self.__verified_regions.append(verified_region)
        self.__record.seq = Seq(mark_verified_region(self.__get_seq()))

    def remove_verified_regions_subsets(self):

        def is_region_subset(region1, region2):
            start1, end1 = region1
            start2, end2 = region2

            return start1 <= start2 and end1 >= end2

        to_remove = set()

        for i in range(len(self.__verified_regions)):
            for j in range(len(self.__verified_regions)):
                if i != j \
                        and i not in to_remove \
                        and is_region_subset(self.__verified_regions[i], self.__verified_regions[j]):
                    to_remove.add(j)

        self.__verified_regions = [
            self.__verified_regions[i] for i in range(len(self.__verified_regions)) if i not in to_remove
        ]

    @property
    def verified_regions(self):
        return self.__verified_regions

    @property
    def record(self):
        return self.__record
