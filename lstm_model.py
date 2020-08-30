

def char_maps():
    """
    Create mapping from the unique chars in pretein sequences to integers and
    vice-versa.
    :param seq: Some group of proteins sequences, that holds all unique characters.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the 'preteins langauge'.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    unique = sorted(["\n", "a", "r", "n", "d", "c", "q", "e", "g", "h", "i", "l", "k", "m", "f", "p", "s", "t", "w", "y", "v"])
    char_to_idx = {unique[i]: i for i in range(len(unique))}
    idx_to_char = {i: unique[i] for i in range(len(unique))}
    
    return char_to_idx, idx_to_char


def upload_sequences(proteins_path):
    with open(proteins_path, 'r') as f:
        proteins_path = f.read()
    return proteins_path.lower(), map(lambda x : x.isupper(), proteins_path)

