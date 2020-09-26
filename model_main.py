import sys
import subprocess
import importlib
import numpy as np
import os
import csv
import re
import xml.etree.ElementTree as et
import pathlib 
import urllib
import urllib.request
import shutil 
import random
import imp
from plot import plot_fit

#===================Our packages====================
import download_data as download
import parse_tcell_epitope as parser
import lstm_model as lstm_model
import train_and_test 
from train_and_test import LSTMTrainer

#=================Constants=========================
TAGSET_SIZE = 2 
#=============================================
url_for_download = "https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip"
out_path = "./samples"

#==============Parameters=====================
#TODO: get those out to config file
#TODO: fill them with the right values
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.01 #learning rate
num_epochs = 500 #the number of times the model will go over the whole dateset
seq_len = 1024
batch_size = 32
hidden_dim = 64
num_layers = 2
bidirectional = False
drop_prop = 0.5

checkpoint_file = 'gdrive/My Drive/LSTMEpitopePredictor/checkpoints/lstm11' #TODO change it!
max_batches = 300
early_stopping = 5

#=================================================
def pip_import_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)

def imports_for_torch():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch import optim
    from torch import Tensor
    import torch.utils.data
    import torch.nn as nn
    import torch.optim as optim

def avg_binary_loss(loss_fn):
  def avg_binary_cross_entropy(predicted, real):
    count_of_1 = real.sum()
    count_of_0 = (1-real).sum()
    if count_of_0 == 0 or count_of_1 == 0:
      return loss_fn()(predicted, real)
    weight = torch.tensor([torch.true_divide(1.0, count_of_0), torch.true_divide(1.0, count_of_1)], device=real.device)
    return loss_fn(weight=weight)(predicted, real)
  return avg_binary_cross_entropy


def post_epoch_fn(epoch, test_res, train_res, verbose):
    # Update learning rate
    scheduler.step(test_res.accuracy)
    # Sample from model to show progress
    if verbose:
        probabilities = lstm_model.get_probabilities(model, subset_text, (char_to_idx,idx_to_char), T=0.1)
        capitalized_text = lstm_model.capitalize(subset_text, probabilities)
        print(capitalized_text)
        
def main():
    pip_import_install("biopython")
    from Bio import Entrez, SeqIO
    imports_for_torch()
    
    print("Downloading data...\n")
    download.download_data(out_path, url_for_download)
    
    print("Organizing data, checking for duplicates (it might take a while... )")
    parser.make_samples(out_path)

    print("Loading the data from memory and creating a division to groups")
    #TODO: take idan's code here and adjust
    #######################################################################
    parser.Clean_id_lines_from_samples(out_path)


    # Create dataset of sequences
    
    char_to_idx, idx_to_char = lstm_model.char_maps()
    vocab_len = len(char_to_idx)

    parsed_clean_antigens = pathlib.Path(out_path, "parsed_clean_epitopes")
    
    samples_list = []
    labels_list = []

    for each_file in parsed_clean_antigens.iterdir():
        #load a batch for one try
        proteins_seq = lstm_model.upload_sequences(str(parsed_clean_antigens.joinpath(each_file.name)))
        # Create labelled samples
        samples, labels = lstm_model.chars_to_labelled_samples(proteins_seq, char_to_idx, seq_len, device)
        print(f'samples shape: {samples.shape}')
        print(f'labels shape: {labels.shape}')
        
  
        print(f'sample 100 as text:\n{lstm_model.onehot_to_chars(samples[100],idx_to_char)}')

        samples_list.append(samples)
        labels_list.append(labels)

    samples = torch.cat(samples_list)
    labels = torch.cat(labels_list)
    del samples_list
    del labels_list
    print(f'samples shape: {samples.shape}')
    print(f'labels shape: {labels.shape}')

    # Create DataLoader returning batches of samples.
    ds_seqs = torch.utils.data.TensorDataset(samples, labels)
    dl_seqs = torch.utils.data.DataLoader(ds_seqs, batch_size=batch_size, shuffle=False)

    print(f'num batches: {len(dl_seqs)}')

    x0, y0 = next(iter(dl_seqs))
    print(f'shape of a batch sample: {x0.shape}')
    print(f'shape of a batch label: {y0.shape}')
    ##########################################################################


    # Full dataset definition
    vocab_len = len(char_to_idx)
    batch_size = 1
    train_test_ratio = 0.8
    num_samples = (len(samples))
    num_train = int(train_test_ratio * num_samples)

    ds_train = torch.utils.data.TensorDataset(samples[:num_train], labels[:num_train])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=True)

    ds_test = torch.utils.data.TensorDataset(samples[num_train:], labels[num_train:])
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f'Train: {len(dl_train):3d} batches, {len(dl_train)*batch_size*seq_len:7d} chars')
    print(f'Test:  {len(dl_test):3d} batches, {len(dl_test)*batch_size*seq_len:7d} chars')

    #============== TRAINING ===================
    model = lstm_model.LSTMTagger(hidden_dim=hidden_dim, input_dim=in_dim, tagset_size=tagset_size, n_layers=n_layers, bidirectional=bidirectional, drop_prob=dropout, device=device)
    
    loss_fn = avg_binary_loss(nn.CrossEntropyLoss)
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    trainer = LSTMTrainer(model, loss_fn, optimizer, device)

    # Train, unless final checkpoint is found
    checkpoint_file_final = f'{checkpoint_file}_final.pt'
    if os.path.isfile(checkpoint_file_final):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        saved_state = torch.load(checkpoint_file_final, map_location=device)
        model.load_state_dict(saved_state['model_state'])
    else:
        try:
            # Print pre-training sampling
            probabilities = lstm_model.get_probabilities(model, subset_text, (char_to_idx,idx_to_char), T=0.1)
            capitalized_text = lstm_model.capitalize(subset_text, probabilities)
            # Stop if we've successfully memorized the small dataset.
            print(capitalized_text)

            fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches,
                                  post_epoch_fn=post_epoch_fn, early_stopping=early_stopping,
                                  checkpoints=checkpoint_file, print_every=1)

            fig, axes = plot_fit(fit_res)

        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')

if __name__ == "__main__":
    main()