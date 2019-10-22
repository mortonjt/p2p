from torch.utils.data import Dataset
import tqdm
import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import glob
import collections
import re


class PeptideDataset(Dataset):

    def __init__(self, fasta_file, mapping_file, interaction_file,
                 num_negative=10):
        seqs = list(SeqIO.read(open(fasta_file, "r"), "fasta"))
        self.mapping = pd.read_csv(mapping_file, sep='\t', header=None)
        self.num_negative = num_negative
        # assumes mapping is in the same order as seqhandle
        # TODO: make sure that this happens. Drop all blanks in mapping
        self.mapping.columns = ['entrez']
        p = re.compile('^[0-9]+$')
        idx = self.mapping.apply(lambda x: bool(p.match(x['entrez'])))
        self.mapping = self.mapping[idx]

        ids = map(lambda s: s.id, seqs)
        self.seqdict = zip(idx, self.seqs)
        self.mapping['accession'] = ids

        self.interactions = pd.read_csv(interaction_file, sep='\t')

    def __len__(self):
        return self.mapping.shape[0]

    def __getitem__(self, i):
        negs = []
        for _ in range(self.num_negative):
            negs.append(self.random_peptide())

    def __iter__(self):
        pass

    def random_peptide(self):
        j = np.random.randint(0, self.__len__())
        mj = self.mapping.iloc[j]['accession']
        return self.seqdict[mj]



