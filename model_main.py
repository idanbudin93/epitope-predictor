import os
import pathlib
import urllib
import random
from plot import plot_fit
from urllib.parse import urlparse
from Bio import Entrez, SeqIO
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

# ===================Our packages====================
import download_data as download
import parse_tcell_epitope as parser
import lstm_model as lstm_model
import train_and_test
from train_and_test import LSTMTrainer
import run_processing
import train_and_test

# =================Constants=========================
TAGSET_SIZE = 2

# parser constants
TCELL_DOWNLOAD_URL = "https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip"
TCELL_CSV_FILENAME = 'tcell_proteins.csv'
out_path = "./samples"
PARSED_SAMPLES_FOLDER_NAME = 'parsed_samples'
BATCH_FILENAME = 'epitope_batch_{batch_number}'
BATCH_FILE_SIZE = 5000  # soft limit
BATCH_REQUEST_SIZE = 25

# proccessing samples
PROCESSED_FOLDER_NAME = "processed"
CLEAN_PROCESSED_SAMPLES = 'processed_clean_samples'

# ==============Parameters=====================
# TODO: get those out to config file
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_file = './checkpoints/lstm_adamw_2_64_0.001'

# model parameters
hidden_dim = 64
n_layers = 2
bidirectional = True
dropout = 0.5

# Train parameters
lr = 0.001  # learning rate
num_epochs = 50  # the number of times the model will go over the whole dateset
early_stopping = 5  # stop after this

# Dataset parameters
seq_len = 1024
batch_size = 1
train_test_ratio = 0.8

# =================================================


def avg_binary_loss(loss_fn):
    def avg_binary_cross_entropy(predicted, real):
        count_of_1 = real.sum()
        count_of_0 = (1-real).sum()
        if count_of_0 == 0 or count_of_1 == 0:
            return loss_fn()(predicted, real)
        weight = torch.tensor([torch.true_divide(1.0, count_of_0), torch.true_divide(
            1.0, count_of_1)], device=real.device)
        return loss_fn(weight=weight)(predicted, real)
    return avg_binary_cross_entropy


def make_labelled_samples(out_path, clean_processed_samples_dir, char_to_idx, idx_to_char, train_test_ratio):
    processed_clean_antigens = pathlib.Path(
        out_path, clean_processed_samples_dir)

    samples_list = []
    labels_list = []

    for each_file in processed_clean_antigens.iterdir():
        # load a batch for one try
        proteins_seq = lstm_model.upload_sequences(
            str(processed_clean_antigens.joinpath(each_file.name)))
        # Create labelled samples
        samples, labels = lstm_model.chars_to_labelled_samples(
            proteins_seq, char_to_idx, seq_len, device)
        print(f'samples shape: {samples.shape}')
        print(f'labels shape: {labels.shape}')

        samples_list.append(samples)
        labels_list.append(labels)

    n_train = int(train_test_ratio*len(samples_list))
    train_samples_list = samples_list[:n_train]
    train_labels_list = labels_list[:n_train]
    test_samples_list = samples_list[n_train:]
    test_labels_list = labels_list[n_train:]

    train_samples = torch.cat(train_samples_list)
    train_labels = torch.cat(train_labels_list)
    test_samples = torch.cat(test_samples_list)
    test_labels = torch.cat(test_labels_list)
    del train_samples_list
    del train_labels_list
    del test_samples_list
    del test_labels_list
    print(f'train_samples shape: {train_samples.shape}')
    print(f'train_labels shape: {train_labels.shape}')
    print(f'test_samples shape: {test_samples.shape}')
    print(f'test_labels shape: {test_labels.shape}')
    return train_samples, train_labels, test_samples, test_labels


def get_dataloaders(train_samples, train_labels, test_samples, test_labels):
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
    return dl_train, dl_test, ds_test


def print_capitalized_model_text(model, text, char_maps):
    probabilities = lstm_model.get_probabilities(
        model, text, char_maps, T=0.1)
    capitalized_text = lstm_model.capitalize(
        text, probabilities)
    print(capitalized_text)


def get_parsed_samples_paths(out_path, parsed_samples_folder_name):
    parsed_samples_path = pathlib.Path(out_path, parsed_samples_folder_name)
    p = parsed_samples_path.glob('**/*')
    return tuple([str(x) for x in p if x.is_file()])

def get_subset_text(ds_seqs, idx_to_char):
    subset_text = ''
    while subset_text.lower() == subset_text:
        # Pick a tiny subset of the dataset
        subset_start = random.randint(0, len(ds_seqs))
        # Convert subset to text
        subset_text = lstm_model.onehot_to_chars(ds_seqs[subset_start][0], idx_to_char)
        subset_text = lstm_model.capitalize_by_labels(subset_text, ds_seqs[subset_start][1])
    print(f'Subset text":\n\n{subset_text}')
    return subset_text

def train_model(model, subset_text, char_maps, dl_train, dl_test):
    loss_fn = avg_binary_loss(nn.CrossEntropyLoss)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    trainer = LSTMTrainer(model, loss_fn, optimizer, device)

    # Train, unless final checkpoint is found
    checkpoint_file_final = f'{checkpoint_file}_final.pt'

    if os.path.isfile(checkpoint_file_final):
        print(
            f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        saved_state = torch.load(checkpoint_file_final, map_location=device)
        model.load_state_dict(saved_state['model_state'])

    else:
        try:
            def post_epoch_fn(epoch, test_res, train_res, verbose):
                # Update learning rate
                scheduler.step(test_res.accuracy)
                # Sample from model to show progress
                if verbose:
                    print_capitalized_model_text(model, subset_text, char_maps)
            fit_res = trainer.fit(dl_train, dl_test, num_epochs,
                                  post_epoch_fn=post_epoch_fn, early_stopping=early_stopping,
                                  checkpoints=checkpoint_file, print_every=1)

            fig, axes = plot_fit(fit_res)
        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')
# ===============================================================================


def main():
    # downloading the data
    print("Downloading data...\n")
    download.download_data(out_path, TCELL_CSV_FILENAME, TCELL_DOWNLOAD_URL)
    downloaded_filename = TCELL_CSV_FILENAME

    # parsing the downloaded data
    print("Organizing data, checking for duplicates (it might take a while...)\n")
    #parser.make_samples(out_path, downloaded_filename, PARSED_SAMPLES_FOLDER_NAME,
    #                    BATCH_FILENAME, BATCH_FILE_SIZE, BATCH_REQUEST_SIZE)
    print("Finished parsing data\n")
    # pre-proecessing the data
    print("Clustering data for train-test independence\n")
    parsed_samples_paths = get_parsed_samples_paths(
        out_path, PARSED_SAMPLES_FOLDER_NAME)
    run_processing.main(['-i', *parsed_samples_paths])
    print("Done clustering\n")
    parser.Clean_id_lines_from_samples(
        out_path, PROCESSED_FOLDER_NAME, CLEAN_PROCESSED_SAMPLES)
    print("Loading the data to memory and partitioning to train and test groups\n")

    # Create dataset of sequences
    char_to_idx, idx_to_char = lstm_model.char_maps()
    vocab_len = len(char_to_idx)

    train_samples, train_labels, test_samples, test_labels = make_labelled_samples(
        out_path, CLEAN_PROCESSED_SAMPLES, char_to_idx, idx_to_char, train_test_ratio)

    # ====================== MODEL AND TRAINING ======================

    # Create DataLoader returning batches of samples.
    dl_train, dl_test, ds_test = get_dataloaders(
        train_samples, train_labels, test_samples, test_labels)
    # init model
    model = lstm_model.LSTMTagger(hidden_dim=hidden_dim, input_dim=vocab_len, tagset_size=TAGSET_SIZE,
                                  n_layers=n_layers, bidirectional=bidirectional, drop_prob=dropout, device=device)
    # get random subset text from test dataset
    subset_text = get_subset_text(ds_test, idx_to_char)
    # see how model works before training at all
    print_capitalized_model_text(
        model, subset_text.lower(), (char_to_idx, idx_to_char))
    # train the model
    train_model(model, subset_text, (char_to_idx,
                                     idx_to_char), dl_train, dl_test)


if __name__ == "__main__":
    main()
