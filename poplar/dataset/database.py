import torch
import glob
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler
from poplar.util import dictionary, check_random_state
import numpy as np
import pandas as pd
from Bio import SeqIO


def clean(x, threshold=1024):
    if len(x.seq) > threshold:
        x.seq = x.seq[:threshold]
        return x
    else:
        return x


def preprocess(seqdict, links):
    """ Preprocesses sequences / links.

    seqdict: dict of seq
       Sequence lookup table

    TODO: Return taxa specific information.
    """
    # 0 = protein 1
    # 1 = protein 2
    pairs = links.apply(
        lambda x: (np.array(seqdict[x[0]].seq),
                   np.array(seqdict[x[1]].seq)),
        axis=1)
    pairs = np.array(list(pairs.values))
    return pairs


def construct_dataloader(
        seqdict, links,
        training_column=4,
        database_name,
        batch_size=10, num_neg=10, num_workers=1,
        arm_the_gpu=False):
    """ Reads in data and creates dataloaders for interactions.

    Parameters
    ----------
    seqdict : dict of Bio.Sequence objects
        Dictionary of protein sequences, with sequence id lookup keys.
    link : pd.DataFrame
        Dataframe specifying training data.
    training_column : int
        Specifies which samples are for training and testing,
        in the links file. These must be labeled as
        'Train', 'Test' or 'Validate'.
    database_name : str
        Name of database.
    batch_size : int
        Number of protein triples to analyze in a given batch.
    num_workers : int
        Number of workers for training (1 worker for testing).
    arm_the_gpu : bool
        Use a gpu or not.

    Returns
    -------
    Dictionary of dataloaders

    TODO
    ----
    Create unittests for this function
    """
    train = links.loc[links[training_column] == 'Train']
    test = links.loc[links[training_column] == 'Test']
    valid = links.loc[links[training_column] == 'Validate']
    sampler = NegativeSampler(seqs)

    res = {}
    for data in [train, test, valid]:
    if data.shape[0] > 0:
        pairs = preprocess(seqdict, data)
        intdata = InteractionDataset(
            pairs, sampler, num_neg=num_neg)
        res[f'{database_name}_Train'] = DataLoader(
            intdata, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            drop_last=True, pin_memory=arm_the_gpu)
    return res
