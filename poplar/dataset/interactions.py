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

def parse_combined_interactions(
        fasta_file, links_file, training_column=2,
        batch_size=10, num_neg=10, num_workers=1,
        arm_the_gpu=False):
    """ Reads in data and creates dataloaders for interactions.

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

    Notes
    -----
    This assumes that the training file is in the tsv format with
    the following columns.

    0. protein 1 id
    1. protein 2 id
    2. database source
    3. taxonomy
    4. train/test/validate
    5. positive / negative (TODO!)

    For validation, it is assumed that the first protein id comes from
    the pathogen and the second protein id comes from the host.

    Questions
    ---------
    Would it be better to force the user to input a single file
    with all of the columns labeled, or have the user input
    separate files?

    Note: we currently force the user to input a directory
    containing multiple files (with the columns labeled).
    The reason for this is because pytorch dataloaders cannot
    handle datasets that are too large. The potential disadvantage is
    that it could get confusing from a users point of view.
    Furthermore, this complicates how the testing/validation should be done.
    The test/validation dataset should be sorted by (1) taxonomy,
    then by (2) protein1.

    Idea: would it be worthwhile to have the user specify a manifest
    for the input files? PROBABLY NOT!

    Would it be beneficial to make preprocessing scripts available?
    DEFINITE YES!

    Idea
    ----
    1. Make scripts for converting string / hpidb to poplar format public.
    2. Make a preprocessing script to combine datasets together.
       This will (1) fragment the training data into multiple pieces and
       (2) merge testing data and sort by taxonomy + protein1 and
       (3) merge validation data and sort by taxonomy + protein1

    TODO
    ----
    Allow for the user to specify if the data is positive or negative.
    """
    seqs = list(SeqIO.parse(fasta_file, format='fasta'))
    # obtain sequences
    truncseqs = list(map(clean, seqs))
    seqids = list(map(lambda x: x.id, truncseqs))
    seqdict = dict(zip(seqids, truncseqs))

    links = pd.read_table(links_file, header=None, index_col=0)
    dbnames = list(links[2].values_counts().index)
    db_loaders = {}
    for db in dbnames:
        db_links = links.loc[links[2] == db]
        db_loader = construct_dataloader(
            seqdict, links,
            training_column=training_column,
            database_name=db,
            batch_size=batch_size,
            num_neg=num_neg,
            num_workers=num_workers,
            arm_the_gpu=False)
        # merge dataloaders
        db_loaders = {**db_loaders, **db_loader}
    return db_loaders


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
    """ Dataset for training and testing. """
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


class ValidationDataset(InteractionDataset):
    """ Dataset for validation.

    Question: Do we even need this class???

    TODO:
    1. Allow for this dataset to return per-protein specific batches.
    2. Clone this dataset class to allow for taxon-specific batches.

    We probably don't need to inherit a dataloader for this class.
    This is because the batch sizes maybe variable (per protein or per taxon).

    This class likely does not need multiple workers either.
    """

    def __init__(self, pos_pairs, neg_pairs, sampler=None, num_neg=10, seed=0):
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
        super().__init__(pos_pairs, sampler, num_neg, seed)
        self.neg_pairs = neg_pairs

    def __getitem__(self, i):
        """ Retrieves protein pairs

        Parameters
        ----------
        gene : str
            Protein of interest
        pos : str
            Positive interacting protein
        neg1 : str
            Random negative protein
        neg2 : str
            Random protein known not to interact with neg1.
        rnd : str
            Random protein

        TODO: this should also specify which species is being evaluated!
        """
        gene = self.pairs[i, 0]
        pos = self.pairs[i, 1]
        rnd = self.random_peptide()

        j = np.random.randint(0, len(self.neg_pairs))

        neg1 = self.neg_pairs[j, 0]
        neg2 = self.neg_pairs[j, 1]

        return (
            ''.join(gene), ''.join(pos),
            ''.join(neg1), ''.join(neg2),
            ''.join(rnd)
        )

    def __iter__(self):
        """ Retrieves an iterable of protein pairs

        Parameters
        ----------
        gene : str
            Protein of interest
        pos : str
            Positive interacting protein
        neg1 : str
            Random negative protein
        neg2 : str
            Random protein known not to interact with neg1.
        rnd : str
            Random protein
        """
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
