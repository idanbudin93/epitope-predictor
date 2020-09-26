from os import path, makedirs
from shutil import rmtree
from typing import List

from docker import DockerClient

from preprocess.clustering import cluster_records_by_identity
from preprocess.datasets_processing import get_epitopes_with_max_verified_regions, \
    split_epitopes_clusters_to_cv_datasets
from model.EpitopesClusters import EpitopesClusters
from model.EpitopesDataset import EpitopesDataset


class Processor:
    """
    The Processing object of the epitope records

    Parameters
    ----------
    docker_client: docker.DockerClient
        Docker client object
    temp_output_dir_path: str
        Path to temporary output directory
    cd_hit_img_id: str
        Pulled Docker Image ID of cd-hit tool
    """
    def __init__(self, docker_client: DockerClient, temp_output_dir_path: str, cd_hit_img_id: str):
        self.__epitopes_dataset: EpitopesDataset = EpitopesDataset([])
        self.__temp_output_dir = temp_output_dir_path
        self.__docker_client = docker_client
        self.__cd_hit_img_id = cd_hit_img_id

        makedirs(self.__temp_output_dir, exist_ok=True)

    def __get_cd_hit_docker_img(self, cd_hit_docker_name: str) -> str:
        if len(self.__docker_client.images.list(cd_hit_docker_name)) == 0:
            self.__docker_client.images.pull(cd_hit_docker_name)

        cd_hit_img_id = self.__docker_client.images.list(cd_hit_docker_name)[0].id

        return cd_hit_img_id

    def load_data(self, fasta_paths_lst: List[str]):
        """
        Loads input epitopes batch files

        Parameters
        ----------
        fasta_paths_lst : List[str]
            List of input epitopes batch files in fasta format
        """
        epitopes_dataset = EpitopesDataset(fasta_paths_lst)
        epitopes_dataset.merge_identical_seqs()
        self.__epitopes_dataset = epitopes_dataset

    def print_stats(self):
        """
        Prints the number of epitope records and number of verified regions
        """
        print('epitope records:', len(self.__epitopes_dataset))
        print('verified regions:', self.__epitopes_dataset.count_verified_regions())

    def __cluster_records_by_identity(self, identity_threshold: float, word_size: int) -> EpitopesClusters:
        epitopes_dataset_path = path.join(self.__temp_output_dir, 'cd-hit-input.fasta')
        self.__epitopes_dataset.write(epitopes_dataset_path)
        output_path_without_extension = path.join(self.__temp_output_dir, 'cd-hit-output')
        cluster_records_by_identity(
            self.__docker_client,
            self.__cd_hit_img_id,
            epitopes_dataset_path,
            output_path_without_extension,
            identity_threshold,
            word_size)
        output_path = path.join(output_path_without_extension) + '.clstr'
        homologs_clusters = EpitopesClusters(output_path, epitopes_dataset_path)
        return homologs_clusters

    def remove_homologs(self, homologs_threshold: float, homologs_clustering_word_size: int):
        """
        Remove homolog epitope records according to the given threshold.
        The homolog records are clustered by cd-hit tool.
        Keeping the records with maximum verified regions.

        Parameters
        ----------
        homologs_threshold : float
            Identity threshold for clustering homolog records
        homologs_clustering_word_size : int
            The word size for identity clustering (see cd-hit guide for details)
        """
        homologs_clusters = self.__cluster_records_by_identity(homologs_threshold, homologs_clustering_word_size)
        no_homologs_epitopes_dataset = get_epitopes_with_max_verified_regions(homologs_clusters)
        self.__epitopes_dataset = no_homologs_epitopes_dataset

    def save_epitopes_cv_dataset(
            self,
            cv_group_path_without_number: str,
            identity_threshold: float,
            word_size: int,
            cv_fold: int):
        """
        Splitting the epitopes records to CV datasets.
        Each pair of records with above the given identity or above are in same dataset.
        Saving each dataset to a fasta file with the dataset number in the end of the file name.

        Parameters
        ----------
        cv_group_path_without_number : str
        identity_threshold : float
            Identity threshold for clustering records for datasets splitting
        word_size : int
            The word size for identity clustering (see cd-hit guide for details)
        cv_fold : int
            The number of CV datasets to split the data to
        """
        def get_epitopes_cv_dataset() -> List[EpitopesDataset]:
            epitopes_dataset_path = path.join(self.__temp_output_dir, 'get-epitopes-cv-dataset-input.fasta')
            self.__epitopes_dataset.write(epitopes_dataset_path)
            cv_groups_clusters = self.__cluster_records_by_identity(
                identity_threshold,
                word_size
            )
            return split_epitopes_clusters_to_cv_datasets(cv_groups_clusters, cv_fold)

        epitopes_cv_datasets = get_epitopes_cv_dataset()
        makedirs(path.dirname(cv_group_path_without_number), exist_ok=True)
        for i in range(len(epitopes_cv_datasets)):
            cv_group_path = cv_group_path_without_number.format(i + 1)
            epitopes_cv_datasets[i].write(cv_group_path)
            print(f'CV dataset #{i + 1}:')
            print('\tpath:', cv_group_path)
            print('\tepitope records:', len(epitopes_cv_datasets[i]))
            print('\tverified regions:', epitopes_cv_datasets[i].count_verified_regions())

    def cleanup(self):
        rmtree(self.__temp_output_dir)