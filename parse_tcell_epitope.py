import os
import csv
import re
import xml.etree.ElementTree as et

from Bio import Entrez, SeqIO

SRC_PATH = '/home/userors/academics/epishop/tcell_full_v3.csv'
DEST_PATH = '/home/userors/academics/epishop/parsed_epitopes/epitope_batch_{batch_number}'
BATCH_FILE_SIZE = 5000  # soft limit
BATCH_REQUEST_SIZE = 25

EPITOPE_SECTION = (9, 24)
EPITOPE_COLUMNS = {
    'epitope page': 0,
    'type': 1,
    'sequence': 2,
    'start position': 3,
    'end position': 4,
    'antigen name': 6,
    'antigen page': 7,
    'protein name': 8,
    'protein page': 9
}

Entrez.api_key = 'cc86c67528c5c05e90be06cace971d287b08'
Entrez.email = 'omershapira@mail.tau.ac.il'
Entrez.tool = 'epitope_parser'


def read_csv(file_path):
    line_number = 0
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            line_number += 1
            yield line_number, line[EPITOPE_SECTION[0]: EPITOPE_SECTION[1]]


def get_protein_id(protein_page):
    id_pattern = r'protein/(?P<pid>[a-zA-Z0-9]+(\.[0-9])?)'
    id_match = re.search(id_pattern, protein_page)
    if id_match:
        return id_match.group('pid')


def iterate_epitopes_batched(path, parsed_epitopes):
    epitope_batch = {}
    for line_num, line in read_csv(path):
        try:
            antigen_id = get_protein_id(line[EPITOPE_COLUMNS['antigen page']])
            epitope_seq = line[EPITOPE_COLUMNS['sequence']].lower()

            if antigen_id and epitope_batch.get(antigen_id, '') != epitope_seq and \
                    not duplicate_epitope(parsed_epitopes, antigen_id, epitope_seq):
                epitope_batch[antigen_id] = {
                    'start': int(line[EPITOPE_COLUMNS['start position']]) - 1,
                    'end': int(line[EPITOPE_COLUMNS['end position']]),
                    'seq': epitope_seq.lower()
                }
        except ValueError:
            print("couldn't parse epitope from line {0}".format(line_num))
        if len(epitope_batch) >= BATCH_REQUEST_SIZE:
            yield epitope_batch
            epitope_batch = {}

    yield epitope_batch


def ncbi_request(id_list):
    response = Entrez.efetch(db='protein', rettype='fasta', retmode='text',
                             id=','.join(id_list))
    return response


def highlight_epitope(antigen_sequence, epitope):
    return antigen_sequence[:epitope['start']].lower() + \
            antigen_sequence[epitope['start']: epitope['end']].upper() + \
            antigen_sequence[epitope['end']:].lower()


def get_fasta_id(long_id):
    id_pattern = r'sp\|(?P<pid>[a-zA-Z0-9]+(\.[0-9])?)\|.*'
    id_match = re.search(id_pattern, long_id)
    if id_match:
        return id_match.group('pid')
    else:
        return long_id


def write_entry(file_handle, sequence, **kwargs):
    entry_description = ["{0}={1}".format(k, v) for k, v in kwargs.items()]
    file_handle.write(">{0}{1}".format(" || ".join(entry_description), "\n"))
    file_handle.write(str(sequence) + "\n\n")


def duplicate_epitope(validation_epitopes, antigen_id, epitope_seq):
    if antigen_id in validation_epitopes:
        return epitope_seq == validation_epitopes[antigen_id]
    return False


def epitope_sequence_validation(epitope_data, antigen_sequence):
    epitope_from_antigen = antigen_sequence[epitope_data['start']:epitope_data['end']].lower()
    epitope_from_data = re.search(r'^[a-z]+', epitope_data['seq'])

    if epitope_from_data:
        epitope_from_data = epitope_from_data.group()
    return epitope_from_data == epitope_from_antigen


def parse_epitope_batch(epitope_batch, output_file, parsed_epitopes):
    ncbi_results = ncbi_request(epitope_batch.keys())
    added_epitopes = 0

    for result in SeqIO.parse(ncbi_results, 'fasta'):
        res_id = get_fasta_id(result.id)
        try:
            not_duplicate = not duplicate_epitope(parsed_epitopes, res_id, epitope_batch[res_id]['seq'])
            sequence_ok = epitope_sequence_validation(epitope_batch[res_id], str(result.seq))

            if not_duplicate and sequence_ok:
                highlighted_sequence = highlight_epitope(
                    result.seq, epitope_batch[res_id])
                write_entry(output_file, highlighted_sequence,
                            id=res_id, name=result.name)

                parsed_epitopes[res_id] = epitope_batch[res_id]['seq']
                added_epitopes += 1
            else:
                print("{0}::{1} failed {2} validation".format(
                    res_id, epitope_batch[res_id]['seq'],
                    'sequence' if not_duplicate else 'duplicate'))
        except KeyError as e:
            print("--[[[{0} (edited to {1}) not in batch {2}".format(
                result.id, res_id, list(epitope_batch.keys())))

    return added_epitopes


if __name__ == "__main__":
    file_entries = 0
    output_filenum = 1
    output_file = open(DEST_PATH.format(batch_number=output_filenum), 'w+')
    parsed_epitopes = dict()

    for epitope_batch in iterate_epitopes_batched(SRC_PATH, parsed_epitopes):
        if file_entries >= BATCH_FILE_SIZE:
            print("Output file {0} full. Writing to new file.".format(output_filenum))
            output_file.close()
            output_filenum += 1
            output_file = open(DEST_PATH.format(batch_number=output_filenum), 'w+')
            file_entries = 0

        file_entries += parse_epitope_batch(epitope_batch, output_file, parsed_epitopes)