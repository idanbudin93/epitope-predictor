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
        print(proteins[7:1234])
        
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
  return 2 if x == '\n' else 1 if x.isupper() else 0

def from_tag(x, tag):
  return '\n' if tag == 2 else x.upper() if tag == 1 else x.lower()
  
def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is 1 if it is uppercase and 0 otherwise.
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
    embedded_text = chars_to_onehot(text, char_to_idx)
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


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        embedded_input = chars_to_onehot(start_sequence, char_to_idx)
        h = None
        for _ in range(n_chars - len(out_text)):
            y, h = model(embedded_input.to(
                dtype=torch.float, device=device).unsqueeze(0), h)
            # take the mean along all batches
            y.squeeze_(dim=0)
            distibution = hot_softmax(y, 1, T)
            next_char_idx = torch.multinomial(distibution, 1)
            next_char_id = next_char_idx[-1].item()
            next_char = idx_to_char[next_char_id]
            out_text += next_char
            embedded_input = chars_to_onehot(next_char, char_to_idx)
    # ========================

    return out_text

class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, n_layers, input_dim, tagset_size):
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_batch: Tensor):
        lstm_out, _ = self.lstm(input_batch)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = functional.log_softmax(tag_space, dim=1)
        return tag_scores