from typing import List, Iterator

from Bio import SeqIO

from models.Epitope import Epitope

CLUSTER_PREFIX = '>'
EPITOPE_ID_PREFIX = '>'
EPITOPE_ID_SUFFIX = '...'


class EpitopesClusters:
    def __init__(self, clstr_file_path: str, records_fasta_path: str):
        self.__epitopes_clusters_lst = self.__parse_clstr_file(clstr_file_path, records_fasta_path)

    def __getitem__(self, index: int) -> List[Epitope]:
        return self.__epitopes_clusters_lst[index]

    def __iter__(self) -> Iterator[List[Epitope]]:
        return  self.__epitopes_clusters_lst.__iter__()

    def __len__(self) -> int:
        return len(self.__epitopes_clusters_lst)

    @staticmethod
    def __parse_clstr_file(clstr_file_path: str, records_fasta_path: str) -> List[List[Epitope]]:

        with open(records_fasta_path) as records_fasta_file:
            records_dict = SeqIO.to_dict(SeqIO.parse(records_fasta_file, 'fasta'))

        epitopes_clusters_lst = []

        with open(clstr_file_path) as epitopes_ids_clusters_file:
            curr_cluster = []
            for line in epitopes_ids_clusters_file.readlines():
                line = line.strip()
                if line != '':
                    # when new cluster found appending the current cluster set and creating new one
                    # if the cluster set is not empty (should occur on first line)
                    if line.startswith(CLUSTER_PREFIX):
                        if len(curr_cluster) > 0:
                            epitopes_clusters_lst.append(curr_cluster)
                            curr_cluster = []
                    else:
                        epitope_id = line.split(EPITOPE_ID_PREFIX)[1].split(EPITOPE_ID_SUFFIX)[0]
                        seq_record = records_dict[epitope_id]
                        epitope = Epitope(seq_record)
                        curr_cluster.append(epitope)

            # adding last cluster ser
            if len(curr_cluster) > 0:
                epitopes_clusters_lst.append(curr_cluster)

        return epitopes_clusters_lst
