import torch
import glob
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler
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
    """
    # 0 = protein 1
    # 1 = protein 2
    pairs = links.apply(
        lambda x: (np.array(seqdict[x[1]].seq),
                   np.array(seqdict[x[2]].seq)),
        axis=1)
    pairs = np.array(list(pairs.values))
    return pairs


def parse(fasta_file, links_file, training_column=2,
          batch_size=10, num_workers=1, arm_the_gpu=False):
    """ Reads in data and creates dataloaders.

    Parameters
    ----------
    fasta_file : filepath
        Fasta file of sequences of interest.
    link_file : filepath
        Table of tab delimited interactions.
    training_column : str
        Specifies which samples are for training and testing,
        in the links file. These must be labeled as
        'Train' and 'Test'.
    batch_size : int
        Number of protein triples to analyze in a given batch.
    num_workers : int
        Number of workers for training (1 worker for testing).
    arm_the_gpu : bool
        Use a gpu or not.
    """
    seqs = list(SeqIO.parse(fasta_file, format='fasta'))
    links = pd.read_table(links_file, header=None, index_col=0)
    train_links = links.loc[links[3] == 'Train']
    test_links = links.loc[links[3] == 'Test']

    # obtain sequences
    truncseqs = list(map(clean, seqs))
    seqids = list(map(lambda x: x.id, truncseqs))
    seqdict = dict(zip(seqids, truncseqs))
    # create pairs
    train_pairs = preprocess(seqdict, train_links)
    test_pairs = preprocess(seqdict, test_links)
    train_dataset = InteractionDataset(train_pairs)
    test_dataset = InteractionDataset(test_pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=arm_the_gpu)

    test_batch_size = max(batch_size, len(test_dataset) - 1)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 drop_last=False, shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=arm_the_gpu)
    return train_dataloader, test_dataloader


class InteractionDataDirectory(Dataset):

    def __init__(self, fasta_file, links_directory,
                 training_column=2,
                 batch_size=10, num_workers=1, arm_the_gpu=False):
        print('links_directory', links_directory)
        self.fasta_file = fasta_file
        self.filenames = glob.glob(f'{links_directory}/*')
        self.training_column = training_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.arm_the_gpu = arm_the_gpu
        self.index = 0

    def __len__(self):
        return len(self.filenames)

    def total(self):
        t = 0
        for fname in self.filenames:
            res = parse(self.fasta_file, fname, self.training_column,
                        self.batch_size, self.num_workers, self.arm_the_gpu)
            # number of sequences in a dataset = (num batch) x (batch size)
            t += len(res[0]) * res[0].batch_size
        return t            
        
    def __iter__(self):
        return (
            parse(self.fasta_file, fname, self.training_column,
                  self.batch_size, self.num_workers, self.arm_the_gpu)
            for fname in self.filenames
        )
 

class InteractionDataset(Dataset):

    def __init__(self, pairs, num_neg=10):
        """ Read in pairs of proteins

        Parameters
        ----------
        pairs: np.array of str
            Pairs of proteins that are experimentally validated to have
            an interaction.
        """
        self.pairs = pairs
        self.num_neg = num_neg

    def random_peptide(self):
        i = np.random.randint(0, len(self.pairs))
        j = int(np.round(np.random.random()))
        return self.pairs[i, j]

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, i):
        gene = self.pairs[i, 0]
        pos = self.pairs[i, 1]
        neg = self.random_peptide()
        return ''.join(gene), ''.join(pos), ''.join(neg)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        w = float(worker_info.num_workers)
        start = 0
        end = len(self.links)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                yield self.__getitem__[i]
        else:
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                yield self.__getitem__[i]
