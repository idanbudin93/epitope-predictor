import argparse
import json
import random
from os import path, getcwd

import docker
from docker import DockerClient

from preprocess.Preprocessor import Processor


def get_args(raw_args=None) -> argparse.Namespace:
    """
    Processing the command arguments

    Returns
    -------
    args : argparse.Namespace
        Arguments Namespace Object
    """
    default_config_path = path.join(getcwd(), 'config.json')
    parser = argparse.ArgumentParser(description='Preprocessing the input epitopes data for the LSTM model')
    parser.add_argument('-i', '--input_files', nargs='+', required=True, help='List of input files in fasta format')
    parser.add_argument('-c', '--config', default=default_config_path,
                        help=f'Config file in json format (default is {default_config_path})')
    parser.add_argument('-s', '--use_rand_seed', action='store_true',
                        help='Flag for using constant random seed (defined in config file)')

    return parser.parse_args(raw_args)


def get_config(config_path: str) -> dict:
    """
    Loading the configurations file

    Parameters
    ----------
    config_path : str
        Path to configurations file

    Returns
    -------
    config : dict
        Dictionary holding the configurations
    """
    with open(config_path) as config_file:
        return json.load(config_file)


def get_cd_hit_docker_img(docker_client: DockerClient, cd_hit_docker_name: str) -> str:
    """
    Pulls cd-hit docker image and returns its Image ID

    Parameters
    ----------
    docker_client : DockerClient
        Docker client
    cd_hit_docker_name : str
        Docker name of cd-hit tool

    Returns
    -------
    cd_hit_img_id : str
        The pulled Docker image ID of cd-hit tool
    """
    if len(docker_client.images.list(cd_hit_docker_name)) == 0:
        docker_client.images.pull(cd_hit_docker_name)

    cd_hit_img_id = docker_client.images.list(cd_hit_docker_name)[0].id

    return cd_hit_img_id


def main(raw_args=None):
    args = get_args(raw_args)
    config = get_config(args.config)

    if args.use_rand_seed:
        random.seed(config['random_seed'])

    docker_client = docker.from_env()

    cd_hit_img_id = get_cd_hit_docker_img(docker_client, config['cd_hit_docker_name'])
    preprocessor = Processor(docker_client, path.abspath(config['temp_output_dir']), cd_hit_img_id)
    input_files = [path.abspath(input_file) for input_file in args.input_files]
    print('loading data...')
    preprocessor.load_data(input_files)
    preprocessor.print_stats()
    print('removing homologs...')
    preprocessor.remove_homologs(config['homologs_threshold'], config['homologs_clustering_word_size'])
    preprocessor.print_stats()
    print('saving epitopes CV datasets...')
    preprocessor.save_epitopes_cv_dataset(
        path.abspath(config['output_template']),
        config['cv_groups_clustering_threshold'],
        config['cv_groups_clustering_word_size'],
        config['cv_fold'])
    print('cleanup...')
    preprocessor.cleanup()


if __name__ == '__main__':
    main()
