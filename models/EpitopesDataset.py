from collections import OrderedDict
from typing import List, Iterator, Union

from Bio import SeqIO

from models.Epitope import Epitope


class EpitopesDataset:
    def __init__(self, records_input: [List[str], List[Epitope]]):
        if all(isinstance(records_batch_fasta_path, str) for records_batch_fasta_path in records_input):
            self.__epitopes = self.__parse_records_batches_fasta_files(records_input)
        if all(isinstance(epitope, Epitope) for epitope in records_input):
            self.__epitopes = records_input

    def __iter__(self) -> Iterator[Epitope]:
        return self.__epitopes.__iter__()

    def __getitem__(self, index: int) -> Epitope:
        return self.__epitopes[index]

    def __len__(self) -> int:
        return len(self.__epitopes)

    def __eq__(self, other):
        return set(self) == set(other)

    @staticmethod
    def __parse_records_batches_fasta_files(records_batches_fasta_paths: List[str]) -> List[Epitope]:
        raw_records = []
        for records_batch_fasta_path in records_batches_fasta_paths:
            with open(records_batch_fasta_path) as records_batch_file:
                records_batch = [Epitope(seq_record) for seq_record in SeqIO.parse(records_batch_file, 'fasta')]
                raw_records.extend(records_batch)

        return raw_records

    def merge_identical_seqs(self):
        merged_epitopes_dict = OrderedDict()

        for epitope in self.__epitopes:
            seq_str = str(epitope)
            seq_key = seq_str.lower()
            verified_region_ind = epitope.verified_regions[0]
            if seq_key in merged_epitopes_dict:
                merged_epitopes_dict[seq_key].add_verified_region(verified_region_ind)
            else:
                merged_epitopes_dict[seq_key] = epitope

        self.__epitopes = list(merged_epitopes_dict.values())

    def count_verified_regions(self) -> int:
        verified_regions_count = 0

        for epitope in self.__epitopes:
            verified_regions_count += len(epitope.verified_regions)

        return verified_regions_count

    def remove_verified_regions_subsets(self):
        for epitope in self.__epitopes:
            epitope.remove_verified_regions_subsets()

    def write(self, output_path):
        with open(output_path, 'w') as output_file:
            fasta_out = SeqIO.FastaIO.FastaWriter(output_file, wrap=None)
            seq_records = [epitope.record for epitope in self.__epitopes]
            fasta_out.write_file(seq_records)
            list({1, 2, 3})
