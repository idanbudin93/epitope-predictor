import os
import pathlib
import urllib
import random
from plot import plot_fit
from urllib.parse import urlparse
from Bio import Entrez, SeqIO
import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import json
# ===================Our packages====================
import download_data as download
import parse_tcell_epitope as parser
import lstm_model as lstm_model
import train_and_test
from train_and_test import LSTMTrainer
from train_and_test import LocalMaximumError
import run_processing

# =================Constants=========================
TAGSET_SIZE = 2
# parser constants
TCELL_DOWNLOAD_URL = "https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip"
TCELL_CSV_FILENAME = 'tcell_proteins.csv'
PARSED_SAMPLES_FOLDER_NAME = 'parsed_samples'
BATCH_FILE_SIZE = 5000  # soft limit
BATCH_REQUEST_SIZE = 25

# proccessing samples
CLEAN_PROCESSED_SAMPLES = 'processed_clean_samples'

# Config file name
CONFIG_FILE = "config.json"
# ==============Globals=====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(CONFIG_FILE) as config_file:
    config = json.load(config_file)["lstm_model_config"]
out_path = config["working_dir"]
# checks for random model initialization
random_uppercase_threshold = config["random_uppercase_threshold"]
# processed samples path
processed_folder_name = config["processed_folder_name"]
# =================================================


def make_labelled_samples(out_path, clean_processed_samples_dir, char_to_idx, idx_to_char, train_test_ratio):
    """
    Reads all already-labeled samples of protein sequences from a given directory,
    splits them to train and test sets according the given raio,
    and returns train and test Tensors of labels and one-hot embedded samples.
    :param out_path: The working directory
    :param clean_processed_samples_dir: The directory inside 
        the working directory that contains the text files of protein sequences.
    :param char_to_idx: a mapping from amino acid letter to id
    :param idx_to_char: a mapping from id to amino acid letter
    :return: a tuple of Tensors:
        - train_samples: the training samples
        - train_labels: the training labels
        - test_samples: the testing samples
        - test_labels: the testing labels
    """
    processed_clean_antigens = pathlib.Path(
        out_path, clean_processed_samples_dir)

    seq_len = config["seq_len"]

    samples_list = []
    labels_list = []

    for each_file in processed_clean_antigens.iterdir():
        # load a batch for one try
        proteins_seq = lstm_model.upload_sequences(
            str(processed_clean_antigens.joinpath(each_file.name)))
        # Create labelled samples
        samples, labels = lstm_model.chars_to_labelled_samples(
            proteins_seq, char_to_idx, seq_len, device)

        samples_list.append(samples)
        labels_list.append(labels)
    # Split the list of protein samples according to the given ratio
    n_train = int(train_test_ratio*len(samples_list))
    train_samples_list = samples_list[:n_train]
    train_labels_list = labels_list[:n_train]
    test_samples_list = samples_list[n_train:]
    test_labels_list = labels_list[n_train:]

    # Concatenate the list of train and test Tensors into one Tensor
    # containing all train and test tensors
    train_samples = torch.cat(train_samples_list)
    train_labels = torch.cat(train_labels_list)
    test_samples = torch.cat(test_samples_list)
    test_labels = torch.cat(test_labels_list)
    del train_samples_list
    del train_labels_list
    del test_samples_list
    del test_labels_list
    return train_samples, train_labels, test_samples, test_labels


def get_dataloaders(train_samples, train_labels, test_samples, test_labels):
    """
    Returns a standard PyTorch Dataset/DataLoader combo for both train and test Tensors.
    :param train_samples: A tensor of embedded train samples
    :param train_labels: A tensor of train labels
    :param test_samples: A tensor of embedded test samples
    :param test_labels: A tensor of test labels
    :return: a tuple of:
        - dl_train: a DataLoader for train set
        - dl_test: a DataLoader for test set
        - ds_train: a TensorDataset for train set
        - ds_test: a TensorDataset for test set
    """
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]

    ds_train = torch.utils.data.TensorDataset(train_samples, train_labels)
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=False, drop_last=True)

    ds_test = torch.utils.data.TensorDataset(test_samples, test_labels)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    print(
        f'Train: {len(dl_train):3d} batches, {len(dl_train)*batch_size*seq_len:7d} chars')
    print(
        f'Test:  {len(dl_test):3d} batches, {len(dl_test)*batch_size*seq_len:7d} chars')
    return dl_train, dl_test, ds_train, ds_test


def get_capitalized_model_text(model, text, char_maps):
    """
    Forward passes the given test through the model to generate
    the predictions distribution, and the capitalizes the text according
    to that distribution.
    :param model: The LSTM model
    :param text: an amino acid string of all lower-case letters
    :param char_maps: a tuple of mapping functions from amino acid letter to id and vice-versa.
    :return: the amino acid string after capitalization according to the model predictions.
    """
    probabilities = lstm_model.get_probabilities(
        model, text, char_maps, T=0.1)
    return lstm_model.capitalize(text, probabilities)


def get_parsed_samples_paths(out_path, parsed_samples_folder_name):
    """
    Returns a tuple with paths of all files in the given folder 
    (i.e. in out_path/parsed_samples_folder_name/)
    """
    parsed_samples_path = pathlib.Path(out_path, parsed_samples_folder_name)
    p = parsed_samples_path.glob('**/*')
    return tuple([str(x) for x in p if x.is_file()])


def get_subset_text(ds_seqs, idx_to_char):
    """
    Selects a random sample from the given dataset that contains epitopes.
    """
    subset_text = ''
    while subset_text.lower() == subset_text:
        # Pick a tiny subset of the dataset
        subset_start = random.randint(0, len(ds_seqs))
        # Convert subset to text
        subset_text = lstm_model.onehot_to_chars(
            ds_seqs[subset_start][0], idx_to_char)
        subset_text = lstm_model.capitalize_by_labels(
            subset_text, ds_seqs[subset_start][1])
    print(f'\nSubset text":\n\n{subset_text}')
    return subset_text


def is_random(text):
    """
    Checks if there are enough lower case and upper case letters in the given text.
    The function check if the portion of lower-case letters and upper-case letters
    is above a given configurable (in config file) threshold.
    """
    if len(text) == 0:
        return False
    upper_p = sum(map(str.isupper, text)) / len(text)
    return upper_p >= random_uppercase_threshold and upper_p <= 1 - random_uppercase_threshold


def train_model(model, subset_text, char_maps, dl_train, dl_test):
    """
    Trains the given LSTM model using the LSTMTrainer class on the given
    train and test DataLoaders.
    :param model: an initialized LSTM model
    :param subset_text: a text that will be printed in each epoch after
        capitalization according to the model predictions.
    :param char_maps: a tuple of mapping functions from amino acid letter to id and vice-versa.
    :param dl_train: a DataLoader for training dataset
    :param dl_test: a DataLoader for testing dataset
    :return: a FitResults object containing the loss, accuracy, fp and fn for each epoch.
    """
    loss_fn = LSTMTrainer.avg_binary_loss(nn.CrossEntropyLoss)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    trainer = LSTMTrainer(model, loss_fn, optimizer, device)
    checkpoint_file = config['checkpoint_file']
    early_stopping = config['early_stopping']

    # Train, unless final checkpoint is found
    checkpoint_file_final = f'{checkpoint_file}_final.pt'
    if os.path.isfile(checkpoint_file_final):
        print(
            f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        saved_state = torch.load(checkpoint_file_final, map_location=device)
        model.load_state_dict(saved_state['model_state'])
    else:
        def post_epoch_fn(epoch, test_res, train_res, verbose):
            # Update learning rate
            scheduler.step(test_res.accuracy)
            # Show progress of the model predictions
            if verbose:
                print(get_capitalized_model_text(
                    model, subset_text, char_maps))
        fit_res = trainer.fit(dl_train, dl_test, config['num_epochs'],
                              post_epoch_fn=post_epoch_fn, early_stopping=early_stopping,
                              checkpoints=checkpoint_file, print_every=1)
        return fit_res

# ===============================================================================


def main():
    # downloading the data
    print("Downloading data...\n")
    download.download_data(out_path, TCELL_CSV_FILENAME, TCELL_DOWNLOAD_URL)
    downloaded_filename = TCELL_CSV_FILENAME

    # parsing the downloaded data
    print("Organizing data, checking for duplicates (it might take a while...)\n")
    parser.make_samples(out_path, downloaded_filename,
                        PARSED_SAMPLES_FOLDER_NAME, BATCH_FILE_SIZE, BATCH_REQUEST_SIZE)
    print("Finished parsing data\n")
    # pre-proecessing the data
    print("Clustering data for train-test independence\n")
    parsed_samples_paths = get_parsed_samples_paths(
        out_path, PARSED_SAMPLES_FOLDER_NAME)
    run_processing.main(['-i', *parsed_samples_paths])
    print("Done clustering\n")
    parser.Clean_id_lines_from_samples(
        out_path, processed_folder_name, CLEAN_PROCESSED_SAMPLES)
    print("Loading the data to memory and partitioning to train and test groups\n")

    # Create dataset of sequences
    char_to_idx, idx_to_char = lstm_model.char_maps()
    vocab_len = len(char_to_idx)

    train_samples, train_labels, test_samples, test_labels = make_labelled_samples(
        out_path, CLEAN_PROCESSED_SAMPLES, char_to_idx, idx_to_char, config["train_test_ratio"])

    # ====================== MODEL AND TRAINING ======================

    # Create DataLoader returning batches of samples.
    dl_train, dl_test, _, ds_test = get_dataloaders(
        train_samples, train_labels, test_samples, test_labels)
    # get random subset text from test dataset
    subset_text = get_subset_text(ds_test, idx_to_char)
    # initialize a model and try to train it
    in_local_maximum = True
    while in_local_maximum:
        try:
            in_local_maximum = False
            model_text = ''
            model = None
            print(
                "\nInitializing a random model with a random enough capitalization before training\n")
            while not is_random(model_text):
                # init model
                model = lstm_model.LSTMTagger(hidden_dim=config["hidden_dim"], input_dim=vocab_len, tagset_size=TAGSET_SIZE,
                                              n_layers=config["n_layers"], bidirectional=(config["bidirectional"] == 1), drop_prob=config["dropout"], device=device)
                model.to(device)
                model_text = get_capitalized_model_text(
                    model, subset_text.lower(), (char_to_idx, idx_to_char))

            # see how model works before training at all
            print("Model capitalization before training:\n")
            print(model_text)
            # train the model
            fit_res = train_model(model, subset_text, (char_to_idx,
                                                       idx_to_char), dl_train, dl_test)
            print("\nFinished training\n")
        except LocalMaximumError:
            print("Stuck in local maximum of all non-epitopes! Retrying...")
            checkpoint_file = config['checkpoint_file']
            checkpoint_filename = f'{checkpoint_file}.pt'
            if os.path.isfile(checkpoint_filename):
                pathlib.Path(checkpoint_filename).unlink()
            in_local_maximum = True
    # plot the training results
    training_plot_name = config['training_plot_name']
    print(f"Saving training plot to: {training_plot_name}\n")
    fig, _ = plot_fit(fit_res)
    fig.savefig(training_plot_name)


if __name__ == "__main__":
    main()
