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
          batch_size=10, num_neg=10, num_workers=1, arm_the_gpu=False):
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
    valid = np.logical_or(links[3] == 'Validate',
                          links[3] == 'Negative')
    valid_links = links.loc[valid]

    # obtain sequences
    truncseqs = list(map(clean, seqs))
    seqids = list(map(lambda x: x.id, truncseqs))
    seqdict = dict(zip(seqids, truncseqs))
    # create pairs
    train_pairs = preprocess(seqdict, train_links)
    test_pairs = preprocess(seqdict, test_links)
    valid_pairs = preprocess(seqdict, valid_links)

    sampler = NegativeSampler(seqs)
    train_dataset = InteractionDataset(train_pairs, sampler, num_neg=num_neg)
    test_dataset = InteractionDataset(test_pairs, sampler, num_neg=num_neg)
    valid_dataset = InteractionDataset(valid_pairs, sampler, num_neg=num_neg)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=arm_the_gpu)

    test_batch_size = max(batch_size, 1)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 drop_last=False, shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=arm_the_gpu)

    valid_batch_size = max(batch_size, 1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size,
                                  drop_last=False, shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=arm_the_gpu)

    return train_dataloader, test_dataloader, valid_dataloader


class NegativeSampler(object):
    """ Sampler for negative data """
    def __init__(self, seqs):
        self.seqs = seqs

    def draw(self):
        """ Draw at random. """
        i = np.random.randint(0, len(self.seqs))
        return self.seqs[i].seq


class InteractionDataDirectory(Dataset):

    def __init__(self, fasta_file, links_directory,
                 training_column=2, num_neg=5,
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
                        self.batch_size, num_neg, self.num_workers,
                        self.arm_the_gpu)
            # number of sequences in a dataset = (num batch) x (batch size)
            t += len(res[0]) * res[0].batch_size
        return t

    def __iter__(self):
        return (
            parse(self.fasta_file, fname, self.training_column,
                  self.batch_size, num_neg, self.num_workers,
                  self.arm_the_gpu)
            for fname in self.filenames
        )


class InteractionDataset(Dataset):

    def __init__(self, pairs, sampler=None, num_neg=10, seed=0):
        """ Read in pairs of proteins

        Parameters
        ----------
        pairs: np.array of str
            Pairs of proteins that are experimentally validated to have
            an interaction.
        sampler : poplar.sample.NegativeSampler
            Model for drawing negative samples for training
        num_neg : int
            Number of negative samples
        seed : int
            Random seed
        """
        self.pairs = pairs
        self.num_neg = num_neg
        self.state = check_random_state(seed)
        self.sampler = sampler
        if sampler is None:
            self.num_neg = 1

    def random_peptide_draw(self):
        i = self.state.randint(0, len(self.pairs))
        j = int(np.round(self.state.random()))
        return self.pairs[i, j]

    def random_peptide_uniform(self):
        # randomly generate a length
        # the average peptide has 300 residues
        l = self.state.poisson(300)
        l = max(30, l)
        l = min(1024, l)

        res = set(dictionary.keys())
        res = list(res - {'.'})
        res = sorted(res)
        seq = self.state.randint(0, len(res), size=l)
        seq = ''.join(list(map(lambda x: res[x], seq)))
        return seq

    def random_peptide(self):
        if self.sampler is None:
            raise ("No negative sampler specified")

        return self.sampler.draw()

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, i):
        gene = self.pairs[i, 0]
        pos = self.pairs[i, 1]
        neg = self.random_peptide()
        return ''.join(gene), ''.join(pos), ''.join(neg)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.pairs)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                for _ in range(self.num_neg):
                    yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)

            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                for _ in range(self.num_neg):
                    yield self.__getitem__(i)
