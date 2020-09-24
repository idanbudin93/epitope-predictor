import argparse
import json
import random
from os import path, getcwd, makedirs

import docker
from docker import DockerClient

from data_processing.clustering import cluster_records_by_identity
from data_processing.datasets_processing import get_epitopes_with_max_verified_regions, \
    split_epitopes_clusters_to_cv_datasets
from model.EpitopesClusters import EpitopesClusters
from model.EpitopesDataset import EpitopesDataset


def get_args():
    default_output_dir = path.join(getcwd(), 'output')
    default_config_path = path.join(getcwd(), 'config.json')
    parser = argparse.ArgumentParser(description='Preprocessing the input epitopes data for the LSTM model')
    parser.add_argument('-i', '--input_files', nargs='+', required=True, help='List of input files in fasta format')
    parser.add_argument('-o', '--output_dir', default=default_output_dir,
                        help=f'Output directory (default is {default_output_dir})')
    parser.add_argument('-c', '--config', default=default_config_path,
                        help=f'Config file in json format (default is {default_config_path})')
    parser.add_argument('-s', '--use_rand_seed', action='store_true',
                        help='Flag for using constant random seed (defined in config file)')

    return parser.parse_args()


def get_config(config_path: str) -> dict:
    with open(config_path) as config_file:
        return json.load(config_file)


def get_cd_hit_docker_img(docker_client: DockerClient, cd_hit_docker_name: str) -> str:
    if len(docker_client.images.list(cd_hit_docker_name)) == 0:
        docker_client.images.pull(cd_hit_docker_name)

    cd_hit_img_id = docker_client.images.list(cd_hit_docker_name)[0].id

    return cd_hit_img_id


def add_num_to_path(file_path: str, num: int) -> str:
    file_path_without_ext, ext = path.splitext(file_path)
    return file_path_without_ext + str(num) + ext


def main():
    args = get_args()
    config = get_config(args.config)
    temp_output_dir = path.abspath(path.join(args.output_dir, config['temp_output_dir_path']))

    if args.use_rand_seed:
        random.seed(config['random_seed'])

    makedirs(args.output_dir, exist_ok=True)
    makedirs(temp_output_dir, exist_ok=True)

    docker_client = docker.from_env()
    cd_hit_img_id = get_cd_hit_docker_img(docker_client, config['cd_hit_docker_name'])

    print('parsing fasta files...')
    epitopes_dataset = EpitopesDataset(args.input_files)
    print(epitopes_dataset.count_verified_regions(), 'verified regions in dataset (including subsets)')

    print('merging epitope records with identical sequence...')
    epitopes_dataset.merge_identical_seqs()
    print(len(epitopes_dataset), 'merged epitope records in dataset')

    print('removing verified region subsets from records...')
    epitopes_dataset.remove_verified_regions_subsets()
    print(epitopes_dataset.count_verified_regions(), 'verified regions in dataset after removing subsets')
    merged_records_path = path.join(temp_output_dir, config['merged_records_fname'])
    print('saving merged epitope records to:', merged_records_path)
    epitopes_dataset.write(merged_records_path)

    print('clustering homolog records with cd-hit...')
    homologs_clusters_path_without_extension = path.join(
        temp_output_dir,
        config['homologs_clusters_fname_without_extension'])
    cd_hit_stdout_and_err_homologs_clustering = cluster_records_by_identity(
        docker_client,
        cd_hit_img_id,
        merged_records_path,
        homologs_clusters_path_without_extension,
        config['homologs_threshold'],
        config['homologs_clustering_word_size']
    )
    print(cd_hit_stdout_and_err_homologs_clustering)

    print('parsing homologs clusters cd-hit output...')
    homologs_clusters_path = path.join(
        temp_output_dir,
        config['homologs_clusters_fname_without_extension']) + '.clstr'
    homologs_clusters = EpitopesClusters(homologs_clusters_path, merged_records_path)
    print('getting epitope records with maximum verified regions from each homologs cluster...')
    no_homologs_epitopes_dataset = get_epitopes_with_max_verified_regions(homologs_clusters)
    print(len(no_homologs_epitopes_dataset), 'merged epitope records in dataset after removing homologs')
    print(no_homologs_epitopes_dataset.count_verified_regions(), 'verified regions in dataset after removing homologs')
    no_homologs_records_path = path.join(temp_output_dir, config['no_homologs_records_fname'])
    print('saving epitope records without homologs to:', no_homologs_records_path)
    no_homologs_epitopes_dataset.write(no_homologs_records_path)

    print('clustering epitope records by identity for splitting to CV groups...')
    cv_groups_clusters_path_without_extension = path.join(
        temp_output_dir,
        config['cv_groups_clusters_fname_without_extension'])
    cd_hit_stdout_and_err_cv_groups_clustering = cluster_records_by_identity(
        docker_client,
        cd_hit_img_id,
        no_homologs_records_path,
        cv_groups_clusters_path_without_extension,
        config['epitopes_clustering_threshold'],
        config['epitopes_clustering_word_size']
    )
    print(cd_hit_stdout_and_err_cv_groups_clustering)

    print('parsing CV groups clusters cd-hit output...')
    cv_groups_clusters_path = cv_groups_clusters_path_without_extension + '.clstr'
    cv_groups_clusters = EpitopesClusters(cv_groups_clusters_path, no_homologs_records_path)
    print('splitting epitopes into CV groups...')
    epitopes_cv_datasets = split_epitopes_clusters_to_cv_datasets(
        cv_groups_clusters,
        config['cv_fold'])

    print('CV groups consists of:')
    for epitopes_dataset in epitopes_cv_datasets:
        print(
            len(epitopes_dataset), 'epitope records with total of',
            epitopes_dataset.count_verified_regions(), 'verified regions')

    cv_group_path_without_number = path.join(args.output_dir, config['cv_group_fname_without_number'])
    print('Saving each CV group to fasta file...')
    for i in range(len(epitopes_cv_datasets)):
        cv_group_path = add_num_to_path(cv_group_path_without_number, i + 1)
        print('saving:', cv_group_path)
        epitopes_dataset.write(cv_group_path)


if __name__ == '__main__':
    main()
