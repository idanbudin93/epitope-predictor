import argparse
from os import mkdir, path

from preprocessing import utils

FASTA_EXTENSIONS = ['.fasta', '.fna', '.ffn', '.faa', '.frn']
DEFAULT_HOMOLOGOUS_THRESHOLD = 0.8
OUTPUT_DIR = 'processed_data'
NO_HOMOLOGS_FILE_SUFFIX = 'no_homologs'
OUTPUT_FNAME = 'processed_sequences'


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description='Preprocessing data from \
                                                     FASTA file to match the \
                                                     model requirments')
    argparser.add_argument('-i', '--fasta', type=str, help='A path to the \
                                                            input raw \
                                                            FASTA file')
    argparser.add_argument('-o', '--output_dir', type=str,
                           default=OUTPUT_DIR,
                           required=False
                           help='A path to the output files directory')
    argparser.add_argument('-t', '--homologs_threshold', type=float,
                           default=DEFAULT_HOMOLOGOUS_THRESHOLD,
                           required=False
                           help='The threshold to define pair of sequences as \
                                 homologs')
    return parser.parse_args()


def main():
    args = parse_args()

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    seqs_dict = utils.read_fasta(args.fasta_path)
    substitution_matrix = utils.get_blosum62_substitution_matrix()
    utils.remove_homologs(seqs_dict, substitution_matrix,
                          args.homologs_threshold)
    no_homologs_path = path.join(args.output_dir, , OUTPUT_FNAME, '.',
                                 NO_HOMOLOGS_FILE_SUFFIX)
    utils.write_fasta(no_homologs_path)


if __name__ == '__main__':
    main()
