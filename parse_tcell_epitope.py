from sys import argv
import os
import pathlib
import csv
import re
import xml.etree.ElementTree as et

from Bio import Entrez, SeqIO

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

# Entrez.api_key = '<<<maintainer eutils api key>>>'
# Entrez.email = 'maintainer@email.com'
Entrez.tool = 'epitope_parser'


def read_csv(file_path):
    """
    Iteratively read a csv file's lines.

    Parameters
    ----------
    file_path
        Path of the csv file.

    Returns
    -------
        An iterator of (line_number, line_text) tuples.
    """
    line_number = 0
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            line_number += 1
            yield line_number, line[EPITOPE_SECTION[0]: EPITOPE_SECTION[1]]


def get_protein_id(protein_page):
    """
    Gets the protein id from a string of the protein's
    URL in NCBI/uniprot/other site.

    Parameters
    ----------
    protein_page
        String linking to the protein's URL.

    Returns
    -------
        Substring indicating protein id.
    """
    id_pattern = r'protein/(?P<pid>[a-zA-Z0-9]+(\.[0-9])?)'
    id_match = re.search(id_pattern, protein_page)
    if id_match:
        return id_match.group('pid')


def iterate_epitopes_batched(path, parsed_epitopes):
    """
    Iteratively runs through the epitope entries at the file
    in path, returning them in batches for quicker execution.
    The function assumes the source of epitope data is a csv in
    a format like IEDB's tcell assay export.

    Parameters
    ----------
    path
        Filepath of csv to extract epitope data from.
    parsed_epitopes
        A collection of already parsed epitopes to check
        duplicates against.
    Returns
    -------
        A dictionary of epitopes indexed by their antigen's id. each epitope
        is represented by a dict containing the epitope's starting and ending
        positions in the antigen, as well as the epitope's sequence.
    """
    epitope_batch = dict()
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
            epitope_batch = dict()

    yield epitope_batch


def ncbi_request(id_list):
    """
    Requests protein sequence files in fasta format
    from NCBI according to a protein id list.
    NOTE: in order to speed NCBI Eutils request rate,
    set up a maintainer email address and Eutils API key
    in this function or at the start of the script (see
    commanted lines starting with Entrez.<parameter> .

    Parameters
    ----------
    id_list
        A list of protein ids.

    Returns
    -------
        NCBI response fasta handle.
    """
    response = Entrez.efetch(db='protein', rettype='fasta', retmode='text',
                             id=','.join(id_list))
    return response


def highlight_epitope(antigen_sequence, epitope):
    """
    Highlights the epitope subsequence in an antigen
    as capital letters.

    Parameters
    ----------
    antigen_sequence
        Antigen sequence of aa characters
    epitope
        A dictionary representing the epitope's starting
        and ending positions in the antigen.

    Returns
    -------
        A sequence identical to antigen_sequence, except
        the epitope subsequence is capitalized.
    """
    return antigen_sequence[:epitope['start']].lower() + \
            antigen_sequence[epitope['start']: epitope['end']].upper() + \
            antigen_sequence[epitope['end']:].lower()


def get_fasta_id(long_id):
    """
    Tries to match a short identifying pattern in
    an NCBI fasta file description line's id part.
    If no match is found, returns the whole long id.

    Parameters
    ----------
    long_id
        fasta sequence id (given by biopython). NCBI's
        sequence id contains a lot of additional info which
        can be trimmed.

    Returns
    -------
        A trimmed protein sequence id
    """
    id_pattern = r'sp\|(?P<pid>[a-zA-Z0-9]+(\.[0-9])?)\|.*'
    id_match = re.search(id_pattern, long_id)
    if id_match:
        return id_match.group('pid')
    else:
        return long_id


def write_entry(file_handle, sequence, **kwargs):
    """
    Writes an annotated antigen sequence to a file handle
    in fasta format, as well as any additional arguments
    provided by user.

    Parameters
    ----------
    file_handle
        File to write antigen sequence to.
    sequence
        Annotated antigen sequence.

    Returns
    -------
        Nothing.
    """
    entry_description = ["{0}={1}".format(k, v) for k, v in kwargs.items()]
    file_handle.write(">{0}{1}".format(" || ".join(entry_description), "\n"))
    file_handle.write(str(sequence) + "\n\n")


def duplicate_epitope(validation_epitopes, antigen_id, epitope_seq):
    """
    Checks whether a reference epitope (represented by epitope_seq and
    antigen_id) is a duplicate of an epitope in validation_epitopes.

    Parameters
    ----------
    validation_epitopes
        Already written epitopes to check for duplicates against.
    antigen_id
        Antigen id of the reference epitope.
    epitope_seq
        Amino acid sequence of the reference epitope.
    Returns
    -------
        True if the reference epitope is a duplicate, false otherwise.
    """
    if antigen_id in validation_epitopes:
        return epitope_seq in validation_epitopes[antigen_id]
    return False


def epitope_sequence_validation(epitope_data, antigen_sequence):
    """
    Checks whether the epitope sequence given at the source file
    matches the antigen sequence provided by NCBI at the epitope's
    positions.

    Parameters
    ----------
    epitope_data
        Epitope start position, end position and sequence.
    antigen_sequence
        Amino acid sequence of an antigen (should contain the epitope).

    Returns
    -------
        True if the epitope sequence matches the antigen sequence
        in the specified positions, otherwise False.
    """
    epitope_from_antigen = antigen_sequence[epitope_data['start']:epitope_data['end']].lower()
    epitope_from_data = re.search(r'^[a-z]+', epitope_data['seq'])

    if epitope_from_data:
        epitope_from_data = epitope_from_data.group()
    return epitope_from_data == epitope_from_antigen


def parse_epitope_batch(epitope_batch, output_file, parsed_epitopes):
    """
    Process and possibly write to file a batch of epitopes, if they
    match the antigen sequence provided by NCBI and are not already
    written to file.

    Parameters
    ----------
    epitope_batch
        Epitope batch to process and possibly add to the
        output_file.
    output_file
        File to write annotated epitopes to.
    parsed_epitopes
        Already parsed epitopes to check duplicates against.

    Returns
    -------
        The number of new epitopes added to output_file
    """
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

                if res_id not in parsed_epitopes:
                    parsed_epitopes[res_id] = set()
                parsed_epitopes[res_id].add(epitope_batch[res_id]['seq'])
                added_epitopes += 1
            else:
                print("{0}::{1} failed {2} validation".format(
                    res_id, epitope_batch[res_id]['seq'],
                    'sequence' if not_duplicate else 'duplicate'))
        except KeyError as e:
            print("--[[[{0} (edited to {1}) not in batch {2}".format(
                result.id, res_id, list(epitope_batch.keys())))

    return added_epitopes


def get_output_path(out_dir, file_id):
    """
    Parameters
    ----------
    out_dir
        directory to store output files in.
    file_id
        incremental id for each output file.

    Returns
    -------
    A path object of the output file.
    """
    return out_dir.joinpath(
        "epitope_batch_{batch_number}".format(batch_number=file_id)
    )


if __name__ == "__main__":
    file_entries = 0
    output_filenum = 1

    # argv[1] should have the path to tcell epitope csv source file.
    source_path = pathlib.Path(argv[1])
    # argv[2] should have the path to a desired output directory.
    dest_path = pathlib.Path(argv[2])

    output_file = open(get_output_path(dest_path, output_filenum), 'w+')
    parsed_epitopes = dict()

    for epitope_batch in iterate_epitopes_batched(source_path, parsed_epitopes):
        if file_entries >= BATCH_FILE_SIZE:
            print("Output file {0} full. Writing to new file.".format(output_filenum))
            output_file.close()
            output_filenum += 1
            output_file = open(get_output_path(dest_path, output_filenum), 'w+')
            file_entries = 0

        file_entries += parse_epitope_batch(epitope_batch, output_file, parsed_epitopes)