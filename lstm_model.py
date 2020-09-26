__author__ = 'Smadar Gazit'

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

def char_maps():
    """
    Create mapping from the unique chars in pretein sequences to integers and
    vice-versa.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the 'preteins langauge'.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    unique = sorted(["\n", "a", "r", "n", "d", "c", "q", "e", "g", "h", "i", "l", "k", "m", "f", "p", "s", "t", "w", "y", "v", "x", "b", "j", "z", "u"])
    char_to_idx = {unique[i]: i for i in range(len(unique))}
    idx_to_char = {i: unique[i] for i in range(len(unique))}
    
    return char_to_idx, idx_to_char

def upload_sequences(proteins_path):
    """
    upload a batch of data to the memory
    :param protein_path: path to a clean batch of proteins
    :returns
    """
    with open(proteins_path, 'r') as f:
        proteins = f.read()
        print(f'Batch length: {len(proteins)} chars')
        
    return proteins


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
 
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = torch.tensor(
        list(map(lambda x: char_to_idx[x], text.lower()))).unsqueeze(1)
    result = torch.zeros([idx.shape[0], len(char_to_idx)],
                         dtype=torch.int8)
    result.scatter_(1, idx, 1)
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    device = embedded_text.get_device() if embedded_text.is_cuda else torch.device('cpu')
    range_tensor = torch.tensor(range(len(idx_to_char)), device=device)
    text_idx = torch.masked_select(range_tensor, embedded_text.to(torch.bool)).tolist()
    result = ''.join(map(lambda x: idx_to_char[x], text_idx))
    return result

def get_tag(x):
  return 1 if x.isupper() else 0
  
def from_tag(x, y):
  return x.upper() if y == 1 else x.lower()

def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # 0. remove samples that do not contain epitopes
    
    # 1. Embed the given text.
    embedded_text = chars_to_onehot(text, char_to_idx)
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    num_samples = len(embedded_text) // seq_len
    embedded_text = embedded_text[:num_samples*seq_len]
    samples = torch.reshape(
        embedded_text, (num_samples, seq_len, len(char_to_idx)))
    samples = samples.to(device=device)
    l = list(map(get_tag, text[:num_samples*seq_len]))
    text_idx = torch.tensor(l, device=device)
    labels = torch.reshape(text_idx, (num_samples, seq_len))
    # ========================
    return samples, labels

def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # ========================
    y = torch.div(y, temperature).to(torch.double)
    maxes = torch.max(y, dim=dim, keepdim=True).values
    result = torch.exp(y - maxes)
    sums = torch.sum(result, dim, keepdim=True, dtype=torch.double)
    result.div_(sums)
    # ========================
    return result


def get_probabilities(model, protein_seq, char_maps, T):
  """
  get the probability for each amino acid to be part of an epitope
  :param model: the LSTM model 
  :param protein_seq: a protein sequence that should by classified
  :param char_maps: function that indices all letters (=amino acids)
  :param model: 
  """
  device = next(model.parameters()).device
  char_to_idx, idx_to_char = char_maps 

  with torch.no_grad(): #meaning: don't train now
    embedded_input = chars_to_onehot(protein_seq, char_to_idx)
    y, _ = model(embedded_input.to(
        dtype=torch.float, device=device).unsqueeze(0))
    # take the mean along all batches
    y.squeeze_(dim=0)
    distribution = hot_softmax(y, 1, T) 
    return distribution


def capitalize_by_labels(text, labels):
  return ''.join(list(map(lambda x,y: from_tag(x,y), text, labels)))


def capitalize(text, distribution):
  """
  hh
  """
  maxind = torch.max(distribution, 1)
  return capitalize_by_labels(text, maxind.indices)

class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, n_layers, input_dim, tagset_size, drop_prob, bidirectional, device):
        """
        input_size – The number of expected features in the input x (at each timestep)
        hidden_size – The number of features in the hidden state h
        num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
        bidirectional – If True, becomes a bidirectional LSTM. Default: False

        :param hidden_dim
        :param n_layers
        :param input_dim
        :param tagset_size
        """
        super(LSTMTagger, self).__init__()
        # The LSTM takes one-hot embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if bidirectional:
          self.multiply_bi = 2
        else:
          self.multiply_bi = 1 

        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*self.multiply_bi, tagset_size)

    def forward(self, input_batch: Tensor, states: Tensor = None):
        #h0 = torch.zeros(self.num_layers*self.multiply_bi, x.size(0), self.hidden_dim).to(self.device)
        #c0 = torch.zeros(self.num_layers*self.multiply_bi, x.size(0), self.hidden_dim).to(self.device)
        #out, (hn, cn) = self.LSTM = (x, (h0,c0))
        #out = self.hidden2tag(out[:, -1, :])
        #out = functional.log_softmax(tag_space, dim=1)

        #return out, (hn, cn)
        lstm_out, (hn, cn) = self.lstm(input_batch, states)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space, (hn, cn)
