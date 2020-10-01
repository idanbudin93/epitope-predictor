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
        integer from zero to the number of unique chars in the 'preteins langauge'=number of amino acids.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """

    unique = sorted(["\n", "a", "r", "n", "d", "c", "q", "e", "g", "h", "i", "l",
                     "k", "m", "f", "p", "s", "t", "w", "y", "v", "x", "b", "j", "z", "u"])
    char_to_idx = {unique[i]: i for i in range(len(unique))}
    idx_to_char = {i: unique[i] for i in range(len(unique))}

    return char_to_idx, idx_to_char


def upload_sequences(proteins_path):
    """
    Upload a batch of data to the memory.
    :param protein_path: path to a clean batch of proteins
    :returns the batch of data that was uploaded to memory
    """

    with open(proteins_path, 'r') as f:
        proteins = f.read()
        print(f'Batch length: {len(proteins)} chars')

    return proteins


def chars_to_onehot(proteins: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of amino-acid-chars as a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param proteins: The amino acid sequence to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where 
    -N is the length of the sequence
    -D is the number of unique chars in the sequence. 
    The dtype of the returned tensor will be torch.int8.
    """

    idx = torch.tensor(
        list(map(lambda x: char_to_idx[x], proteins.lower()))).unsqueeze(1)
    result = torch.zeros([idx.shape[0], len(char_to_idx)],
                         dtype=torch.int8)
    result.scatter_(1, idx, 1)
    return result


def onehot_to_chars(embedded_seq: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_seq: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """

    device = embedded_seq.device
    range_tensor = torch.tensor(range(len(idx_to_char)), device=device)
    text_idx = torch.masked_select(
        range_tensor, embedded_seq.to(torch.bool)).tolist()
    result = ''.join(map(lambda x: idx_to_char[x], text_idx))
    return result


def onehot_to_idx(embedded_seq: Tensor, idx_range: int) -> Tensor:
    """
    Partially reverses the embedding of a text sequence, producing a sequence
    of one-hot embedding indices.
    :param embedded_seq: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :return: A tensor of shape N where each element is the one-hot index
    of a character.
    """

    device = embedded_seq.device
    range_tensor = torch.tensor(range(idx_range), device=device)
    text_idx = torch.masked_select(
        range_tensor, embedded_seq.to(torch.bool)).tolist()
    return text_idx


def get_tag(x):
    """
    :param x: an amino acid letter, in lowercase or uppercase.
    :return: 1 if this amino acid is inside an epitope, else 0.
    """

    return 1 if x.isupper() else 0


def from_tag(x, y):
    """
    for a lowercase letter x, choose if should be an uppercase or not based on y. 
    :param x: amino acid lowercase letter
    :param y: tagging of the letter, 1 if it is inside an epitope or else 0. 
    :return: the amino acid letter with the right capitaliztion
    """

    return x.upper() if y == 1 else x.lower()


def chars_to_labelled_samples(amino_acid_seq: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    :param amino_acid_seq: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where 
    - N is the number of created samples. 
    - S is the seq_len.
    - V is the embedding dimension.
    """

    # Embed the given text
    embedded_seq = chars_to_onehot(amino_acid_seq, char_to_idx)
    # Create the samples tensor by splitting to groups of seq_len:
    num_samples = len(embedded_seq) // seq_len
    embedded_seq = embedded_seq[:num_samples*seq_len]
    samples = torch.reshape(
        embedded_seq, (num_samples, seq_len, len(char_to_idx)))
    samples = samples.to(device=device)
    # Create the labels tensor in a similar way and convert to indices:
    l = list(map(get_tag, amino_acid_seq[:num_samples*seq_len]))
    text_idx = torch.tensor(l, device=device)
    labels = torch.reshape(text_idx, (num_samples, seq_len))
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

    y = torch.div(y, temperature).to(torch.double)
    maxes = torch.max(y, dim=dim, keepdim=True).values
    result = torch.exp(y - maxes)
    sums = torch.sum(result, dim, keepdim=True, dtype=torch.double)
    result.div_(sums)
    return result


def get_probabilities(model, protein_seq, char_maps, T):
    """
    Returns the probability for each amino acid to be part of an epitope.
    :param model: the current LSTM model
    :param protein_seq: a protein sequence to be classified
    :param char_maps: a tuple of mapping functions from amino acid letter to id and vice-versa.
    :param T: temperature for the hot_softmax function.
    :return: A tensor of dimensions [2, sequence length] 
        representing for each letter, its probabilty to be inside an epitope (and not to be in one).
    """

    device = next(model.parameters()).device
    char_to_idx, _ = char_maps

    with torch.no_grad():  # meaning: don't train now
        embedded_input = chars_to_onehot(protein_seq, char_to_idx)
        y, _ = model(embedded_input.to(
            dtype=torch.float, device=device).unsqueeze(0))
        y.squeeze_(dim=0)
        distribution = hot_softmax(y, 1, T)
        return distribution


def capitalize_by_labels(amino_acid_seq, labels):
    """
    Capitalize a lowercase amino acid sequence according to the label of each letter.
    :param amino_acid_seq: lowercase amino acid sequence
    :param labels: for each letter, 1 if it is in epitope or else 0
    :return: the capitalized sequnce according to the labels 
    """
    return ''.join(list(map(lambda x, y: from_tag(x, y), amino_acid_seq, labels)))


def capitalize(amino_acid_seq, distribution):
    """
    Capitalize a lowercase amino acid sequence according to the given distribution.
    For each letter, capitalize it if it is more probable according to the distribution.
    :param amino_acid_seq: a sequence of amino acids in lowercase letters.  
    :param distribution: for each amino acid, it's probabillity to be in an epitope and not to be in one. 
    :return: the capitalized sequence according to the distribution
    """
    maxind = torch.max(distribution, 1)
    return capitalize_by_labels(amino_acid_seq, maxind.indices)


class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, n_layers, input_dim, tagset_size, drop_prob, bidirectional, device, embedding_dim=None):
        """
        :param hidden_dim: The number of features in the hidden state h
        :param n_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean
                     stacking two LSTMs together to form a stacked LSTM, with the second 
                     LSTM taking in outputs of the first LSTM and computing the final results.
        :param input_dim: The number of expected features in the input x (at each timestep)
        :param tagset_size: the size of tagging space (=how many labels). 
        :param drop_prob: If non-zero, introduces a Dropout layer on the outputs 
                          of each LSTM layer except the last layer, with dropout probability
                          equal to drop_prob.
        :param bidirectional: if set to True, the LSTM model is bidirectional. 
        :param device: the device to work on (i.e. 'cpu', 'cuda'). 
        :param embedding_dim: output dimension of the embedding layer. 
                              Can be set to None to disable word embedding.
                              Default: None.
        """
        super(LSTMTagger, self).__init__()

        if bidirectional:
            self.multiply_bi = 2
        else:
            self.multiply_bi = 1

        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.device = device
        self.dropout = drop_prob
        lstm_dim = input_dim

        # Embedding layer
        self.embedding_dim = embedding_dim
        if embedding_dim:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            lstm_dim = embedding_dim
        # LSTM layer
        self.lstm = nn.LSTM(lstm_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=drop_prob, bidirectional=bidirectional)
        # Linear layer from hidden to tag space
        self.hidden2tag = nn.Linear(hidden_dim*self.multiply_bi, tagset_size)

    def forward(self, input_batch: Tensor, states: Tensor = None):
        """
        Implements the forward pass for the model
        :param input_batch: batch of amino acid sequences
        :param states: a tuple of hidden states and cell states
        :return: A tuple:
        - tag space: for each letter, it's score for each of the possible labels
        - (hn, cn) a tuple of hn - hidden state and cn - cell state 
        """
        lstm_batch = input_batch
        # word embedding of the amino acids
        if self.embedding_dim:
            embed_batch = torch.tensor([
                onehot_to_idx(seq, 26) for seq in input_batch
            ], dtype=torch.long, device=self.device)
            lstm_batch = self.embedding(embed_batch)

        # initialize the hidden states randomly
        if states == None:
            states = self.init_hidden(input_batch.shape[0], self.device)

        # run the lstm layer
        lstm_out, (hn, cn) = self.lstm(lstm_batch, states)

        # map the hidden states to tag space
        tag_space = self.hidden2tag(lstm_out)
        return tag_space, (hn, cn)

    def init_hidden(self, batch_size, device):
        """
        Randomly initializes a hidden states tuple
        :param batch_size: the batch size
        :param device: device to work on (i.e. 'cuda' or 'cpu')
        :return: tuple of randomly initialized hidden states and cell states
        """
        return (torch.rand(self.num_layers*self.multiply_bi, batch_size, self.hidden_dim, device=device),
                torch.rand(self.num_layers*self.multiply_bi, batch_size, self.hidden_dim, device=device))
