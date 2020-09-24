import glob
import json
import os
import random
from os import path

import docker
from docker import DockerClient

from data_processing.clustering import cluster_records_by_identity
from data_processing.datasets_processing import get_epitopes_with_max_verified_regions, \
    split_epitopes_clusters_to_cv_datasets
from model.EpitopesClusters import EpitopesClusters
from model.EpitopesDataset import EpitopesDataset

# todo: move to config and args
DATA_DIR = path.abspath('data')
EPITOPES_BATCH_FNAME_PATTERN = 'epitope_batch_[0-9].fasta'
CD_HIT_DOCKER_NAME = 'quay.io/biocontainers/cd-hit:4.8.1--h8b12597_3'
OUTPUT_DIR = path.abspath('output')

HOMOLOGS_THRESHOLD = 0.8
HOMOLOGS_CLUSTERING_WORD_SIZE = 5
EPITOPES_CLUSTERING_THRESHOLD = 0.5
EPITOPES_CLUSTERING_WORD_SIZE = 3
CV_FOLD = 10
RAND_SEED = 0
IS_DETERMINISTIC = True


TEMP_OUTPUT_DIR = path.abspath('output\\temp')
MERGED_RECORDS_FNAME = 'all_records.merged.fasta'
HOMOLOGS_CLUSTERS_FNAME_WITHOUT_CLSTR_EXT = 'homologs'
HOMOLOGS_CLUSTERS_FNAME = HOMOLOGS_CLUSTERS_FNAME_WITHOUT_CLSTR_EXT + '.clstr'
NO_HOMOLOGS_RECORDS_FNAME = 'all_records.merged.no-homologs.fasta'
CV_GROUPS_CLUSTERS_FNAME_WITHOUT_CLSTR_EXT = 'cv-groups'
CV_GROUPS_CLUSTERS_FNAME = CV_GROUPS_CLUSTERS_FNAME_WITHOUT_CLSTR_EXT + '.clstr'

EPITOPES_BATCH_PATH_PATTERN = path.join(DATA_DIR, EPITOPES_BATCH_FNAME_PATTERN)
MERGED_RECORDS_PATH = path.join(TEMP_OUTPUT_DIR, MERGED_RECORDS_FNAME)
HOMOLOGS_CLUSTERS_PATH_WITHOUT_CLSTR_EXT = path.join(TEMP_OUTPUT_DIR, HOMOLOGS_CLUSTERS_FNAME_WITHOUT_CLSTR_EXT)
HOMOLOGS_CLUSTERS_PATH = path.join(TEMP_OUTPUT_DIR, HOMOLOGS_CLUSTERS_FNAME)
NO_HOMOLOGS_RECORDS_PATH = path.join(TEMP_OUTPUT_DIR, NO_HOMOLOGS_RECORDS_FNAME)
CV_GROUPS_CLUSTERS_PATH_WITHOUT_CLSTR_EXT = path.join(TEMP_OUTPUT_DIR, CV_GROUPS_CLUSTERS_FNAME_WITHOUT_CLSTR_EXT)
CV_GROUPS_CLUSTERS_PATH = path.join(TEMP_OUTPUT_DIR, CV_GROUPS_CLUSTERS_FNAME)


def get_cd_hit_docker_img(docker_client: DockerClient, cd_hit_docker_name: str) -> str:
    if len(docker_client.images.list(CD_HIT_DOCKER_NAME)) == 0:
        docker_client.images.pull(cd_hit_docker_name)

    cd_hit_img_id = docker_client.images.list(CD_HIT_DOCKER_NAME)[0].id

    return cd_hit_img_id


# TODO: add config file to args
def get_config() -> dict:
    curr_dir = path.dirname(path.realpath(__file__))
    config_path = path.join(curr_dir, 'config.json')
    with open(config_path) as config_file:
        return json.load(config_file)


def main():
    if IS_DETERMINISTIC:
        random.seed(RAND_SEED)

    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    docker_client = docker.from_env()
    cd_hit_img_id = get_cd_hit_docker_img(docker_client, CD_HIT_DOCKER_NAME)

    epitopes_batches_paths = glob.glob(EPITOPES_BATCH_PATH_PATTERN)

    print('parsing fasta files...')
    epitopes_dataset = EpitopesDataset(epitopes_batches_paths)
    print(epitopes_dataset.count_verified_regions(), 'verified regions in dataset (including subsets)')

    print('merging epitope records with identical sequence...')
    epitopes_dataset.merge_identical_seqs()
    print(len(epitopes_dataset), 'merged epitope records in dataset')

    print('removing verified region subsets from records...')
    epitopes_dataset.remove_verified_regions_subsets()
    print(epitopes_dataset.count_verified_regions(), 'verified regions in dataset after removing subsets')

    print('saving merged epitope records to:', MERGED_RECORDS_PATH)
    epitopes_dataset.write(MERGED_RECORDS_PATH)

    print('clustering homolog records with cd-hit...')
    cd_hit_stdout_and_err_homologs_clustering = cluster_records_by_identity(
        docker_client,
        cd_hit_img_id,
        MERGED_RECORDS_PATH,
        HOMOLOGS_CLUSTERS_PATH_WITHOUT_CLSTR_EXT,
        HOMOLOGS_THRESHOLD,
        HOMOLOGS_CLUSTERING_WORD_SIZE
    )
    print(cd_hit_stdout_and_err_homologs_clustering)

    print('parsing homologs clusters cd-hit output...')
    homologs_clusters = EpitopesClusters(HOMOLOGS_CLUSTERS_PATH, MERGED_RECORDS_PATH)
    print('getting epitope records with maximum verified regions from each homologs cluster...')
    no_homologs_epitopes_dataset = get_epitopes_with_max_verified_regions(homologs_clusters)
    print(len(no_homologs_epitopes_dataset), 'merged epitope records in dataset after removing homologs')
    print(no_homologs_epitopes_dataset.count_verified_regions(), 'verified regions in dataset after removing homologs')

    print('saving epitope records without homologs to:', NO_HOMOLOGS_RECORDS_PATH)
    no_homologs_epitopes_dataset.write(NO_HOMOLOGS_RECORDS_PATH)

    print('clustering epitope records by identity for splitting to CV groups')
    cd_hit_stdout_and_err_cv_groups_clustering = cluster_records_by_identity(
        docker_client,
        cd_hit_img_id,
        NO_HOMOLOGS_RECORDS_PATH,
        CV_GROUPS_CLUSTERS_PATH_WITHOUT_CLSTR_EXT,
        EPITOPES_CLUSTERING_THRESHOLD,
        EPITOPES_CLUSTERING_WORD_SIZE
    )
    print(cd_hit_stdout_and_err_cv_groups_clustering)
    print('parsing CV groups clusters cd-hit output...')
    cv_groups_clusters = EpitopesClusters(CV_GROUPS_CLUSTERS_PATH, NO_HOMOLOGS_RECORDS_PATH)
    print('splitting epitopes into CV groups...')
    epitopes_cv_datasets = split_epitopes_clusters_to_cv_datasets(
        cv_groups_clusters,
        CV_FOLD
    )
    print('CV groups consists of:')
    for epitopes_dataset in epitopes_cv_datasets:
        print(len(epitopes_dataset), 'epitope records with total of', epitopes_dataset.count_verified_regions(), 'verified regions')


if __name__ == '__main__':
    main()
