__author__ = 'Idan Budin'

import argparse
from os import mkdir, path

from biotite.sequence.align import SubstitutionMatrix

from model.dataset import Dataset

FASTA_EXTENSIONS = ['.fasta', '.fna', '.ffn', '.faa', '.frn']
DEFAULT_HOMOLOGOUS_THRESHOLD = 0.8
OUTPUT_DIR = 'processed_data'
NO_HOMOLOGS_FILE_SUFFIX = 'no_homologs'
OUTPUT_FNAME = 'processed_sequences'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocessing data from FASTA file to match the model requirements')
    parser.add_argument('-i', '--fasta', type=str, help='A path to the input raw FASTA file')
    parser.add_argument('-o', '--output_dir', type=str,
                        default=OUTPUT_DIR,
                        required=False,
                        help='A path to the output files directory')
    parser.add_argument('-t', '--homologs_threshold', type=float,
                        default=DEFAULT_HOMOLOGOUS_THRESHOLD,
                        required=False,
                        help='The threshold to define pair of sequences as homologs')

    return parser.parse_args()


def main():
    args = parse_args()

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    dataset = Dataset(args.fasta_path)
    substitution_matrix = SubstitutionMatrix.std_protein_matrix()
    dataset.remove_homologs(substitution_matrix, args.homologs_threshold)
    no_homologs_path = path.join(args.output_dir, OUTPUT_FNAME, '.', NO_HOMOLOGS_FILE_SUFFIX)
    dataset.write_fasta(no_homologs_path)


if __name__ == '__main__':
    main()
