import torch
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from Bio import SeqIO


def parse(fasta_file, links_file, training_column='Training',
          batch_size=10, num_workers=1, arm_the_gpu=False):
    """ Reads in data and creates dataloaders.

    Parameters
    ----------
    fasta_file : filepath
        Fasta file of sequences of interest.
    link_file : filepath
        Table of tab delimited interactions.
    training_column : str
        Specifies which samples are for training, testing and validation,
        in the links file. These must be labeled as
        'Train', 'Test' and 'Validate'.
    batch_size : int
        Number of protein triples to analyze in a given batch.
    num_workers : int
        Number of workers for training (1 worker for testing and
        another for validation).
    arm_the_gpu : bool
        Use a gpu or not.
    """
    seqs = list(SeqIO.parse(fasta_file, format='fasta'))
    links = pd.read_table(links_file)
    train_links = links.loc[links['Training'] == 'Train']
    test_links = links.loc[links['Training'] == 'Test']
    valid_links = links.loc[links['Training'] == 'Validate']
    train_dataset = InteractionDataset(seqs, train_links)
    test_dataset = InteractionDataset(seqs, test_links)
    valid_dataset = InteractionDataset(seqs, valid_links)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=arm_the_gpu)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=arm_the_gpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=arm_the_gpu)
    return train_dataloader, test_dataloader, valid_dataloader

def clean(x, threshold=1024):
    if len(x.seq) > threshold:
        x.seq = x.seq[:threshold]
        return x
    else:
        return x

class InteractionDataset(Dataset):

    def __init__(self, seqs, links, num_neg=10):
        """ Read in fasta file

        Parameters
        ----------
        seqs : list of Bio.Sequence
            Sequences of interest.
        links : pd.DataFrame
            Table interactions.
        num_neg : int
            Number of negative samples.
        """
        self.links = links

        # truncate sequences to fit
        truncseqs = list(map(clean, seqs))
        
        self.seqids = list(map(lambda x: x.id, truncseqs))
        self.seqdict = dict(zip(self.seqids, truncseqs))

    def random_peptide(self):
        i = np.random.randint(0, len(self.seqids))
        id_ = self.seqids[i]
        return str(self.seqdict[id_].seq)

    def __len__(self):
        return self.links.shape[0]

    def __getitem__(self, i):
        geneid = self.links.iloc[i]['protein1']
        posid = self.links.iloc[i]['protein2']
        neg = self.random_peptide()
        gene = str(self.seqdict[geneid].seq)
        pos = str(self.seqdict[posid].seq)

        
        return gene, pos, neg

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
