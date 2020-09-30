import lstm_model
import train_and_test
import os
import torch
PATH_TO_MODEL_FILE = 'saved_model'
TAGSET_SIZE = 2
#needs to have the model
#model = 
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
    return lstm_model.get_probabilities(model, antigen, (char_to_idx, idx_to_char), T=1)