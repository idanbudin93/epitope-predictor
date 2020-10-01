import random
import os
import pathlib

import click
from Bio import SeqIO

from interface_plot import plot_hot_cold

import lstm_model


import os
import torch

PATH_TO_MODEL_FILE = 'saved_model.pt'
TAGSET_SIZE = 2


def get_probabilities(antigen):
    """
    calculates for each amino acid its probabilty to be inside an epitope,
     according to the trained model
     :param antigen: amino acid seq
     :return 
    """
    if not os.path.isfile(PATH_TO_MODEL_FILE):
        raise RuntimeError(f'Could not find model file {PATH_TO_MODEL_FILE}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_state = torch.load(PATH_TO_MODEL_FILE, map_location=device)
    # model parameters
    hidden_dim = saved_state['hidden_dim']
    n_layers = saved_state['n_layers']
    bidirectional = saved_state['bidirectional']
    dropout = saved_state['dropout']
    char_to_idx, idx_to_char = lstm_model.char_maps()
    vocab_len = len(char_to_idx)
    model = lstm_model.LSTMTagger(hidden_dim=hidden_dim, input_dim=vocab_len, tagset_size=TAGSET_SIZE,
                                  n_layers=n_layers, bidirectional=bidirectional, drop_prob=dropout, device=device)
    model.load_state_dict(saved_state['model_state'])
    distribution = lstm_model.get_probabilities(
        model, antigen, (char_to_idx, idx_to_char), T=1)
    dist_list = [float(prob[1]) for prob in distribution]

    return dist_list


def get_output_dir(parent_dir):
    """
    Tries to create a directory called "prediction_output" inside parent dir. 
    If such file already exists, this function will incrementally try to create
    "prediction_output{N}" directory, alerting the user about the newly created
    directory. 
    :param parent_dir: parent directory for the output folder.
    :return: path to the output directory.
    """
    base_out = parent_dir.joinpath("prediction_output")
    if not base_out.exists():
        os.mkdir(base_out)
        return base_out

    num_out = 1
    num_out_path = parent_dir.joinpath(f"prediction_output{num_out}")
    while num_out_path.exists():
        num_out += 1
        num_out_path = parent_dir.joinpath(f"prediction_output{num_out}")

    print(f"Input directory already has folder/file named prediction_output,"
          f"saving output to {num_out_path} instead.")
    os.mkdir(num_out_path)
    return num_out_path


def add_annotated_record(out_file, seq, probs, desc, threshold):
    annotated_seq = [c.upper() if p > threshold else c for c,
                     p in zip(seq, probs)]
    out_file.write(f">{desc}\n")
    out_file.write(f"{''.join(annotated_seq)}\n\n")


@click.command('predict sequences')
@click.argument('seq_file')
@click.option('--visualize/--no-visualize', default=False, help='create a graph of the predicted probability.')
@click.option('--annotate/--no-annotate', default=False, help='create an annotated file.')
@click.option('--threshold', default=0.9, help='minimal probability for an amino acid to be predicted '
              'as part of an epitope, influances annotation and visualization. '
              'value should be between 0 and 1 (default 0.9).')
@click.option('--show-thres/--hide-thres', default=False, help='show threshold line in graphs.')
def predict_sequences(seq_file, visualize, annotate, threshold, show_thres):
    """
    Epitope Predictor

    authors: Idan Budin, Smadar Gazit, Omer Shapira

    Given a fasta file of amino acid (aa) sequences, this program uses a trained LSTM model to
    predict the probability of each aa sequence to be a part of an epitope.

    All of the program's output is saved to a folder named 'prediction output', in the same
    directory as the input file, and comes in 3 flavors:

     - Probability vectors for each aa sequence, written to a file with the same name as the
      aa sequence.

     - Graphs of prediction probability by position, saved as PNG files with the same name 
      as the aa sequence.

     - A file similar to the input fasta file, only each aa with prediction probability above
      a certain threshold is capitalized.

    WARNING: UNNAMED SEQUENCES OR MULTIPLE SEQUENCES WITH THE SAME NAME MIGHT RESULT IN AN ERROR. 
    """
    seq_path = pathlib.Path(seq_file)
    if not seq_path.exists():
        print(f"File {seq_path} not found. Exiting...")
        return 1

    output_path = get_output_dir(seq_path.parents[0])

    print(f'Parsing {seq_file} with the following options:')
    print(
        f'visualize={visualize}, annotate={annotate}, threshold={threshold}, show threshold={show_thres}')

    with open(seq_file, 'r') as seq_handle:
        if annotate:
            annotated_file = open(output_path.joinpath(
                "annotated_output.fasta"), 'w+')

        for record in SeqIO.parse(seq_handle, 'fasta'):
            probs = get_probabilities(record.seq)

            out_probs = open(output_path.joinpath(
                "{0}.prob".format(record.description)), 'w+')
			
            out_probs.write('\n'.join([str(p) for p in probs]))
            out_probs.close()

            if visualize:
                plot_hot_cold(probs, save_path=output_path.joinpath(record.description),
                              threshold=threshold, plot_threshold=show_thres)

            if annotate:
                add_annotated_record(annotated_file, str(record.seq),
                                     probs, record.description, threshold)

        if annotate:
            annotated_file.close()


if __name__ == '__main__':
    predict_sequences()
