import os
import inspect
import torch
import numpy as np
import numbers


def get_data_path(fn, subfolder='data'):
    """Return path to filename ``fn`` in the data folder.
    During testing it is often necessary to load data files. This
    function returns the full path to files in the ``data`` subfolder
    by default.
    Parameters
    ----------
    fn : str
        File name.
    subfolder : str, defaults to ``data``
        Name of the subfolder that contains the data.
    Returns
    -------
    str
        Inferred absolute path to the test data for the module where
        ``get_data_path(fn)`` is called.
    Notes
    -----
    The requested path may not point to an existing file, as its
    existence is not checked.

    This is the same method as borrowed from scikit-bio
    """
    # getouterframes returns a list of tuples: the second tuple
    # contains info about the caller, and the second element is its
    # filename
    callers_filename = inspect.getouterframes(inspect.currentframe())[1][1]
    path = os.path.dirname(os.path.abspath(callers_filename))
    data_path = os.path.join(path, subfolder, fn)
    return data_path


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Note
    ----
    This is from sklearn
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)



dictionary = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,
    ".": 27
}


def encode(x):
    """ Convert string to tokens. """
    tokens = list(map(lambda i: dictionary[i], list(x)))
    tokens = torch.Tensor(tokens)
    tokens = tokens.long()
    return tokens


def tokenize(gene, pos, neg, model, device, pad=1024):

    # extract features, and take <CLS> token
    g = list(map(lambda x: model.extract_features(encode(x))[:, 0, :], gene))
    p = list(map(lambda x: model.extract_features(encode(x))[:, 0, :], pos))
    n = list(map(lambda x: model.extract_features(encode(x))[:, 0, :], neg))

    g_ = torch.cat(g, 0)
    p_ = torch.cat(p, 0)
    n_ = torch.cat(n, 0)

    return g_, p_, n_
