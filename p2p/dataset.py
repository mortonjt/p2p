import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Bio import SeqIO


class InteractionDataset(Dataset):

    def __init__(self, fasta_file, links_file, num_neg=10):
        """ Read in fasta file

        Parameters
        ----------
        fasta_file : filepath
            Fasta file of sequences of interest.
        link_file : filepath
            Table of tab delimited interactions
        """
        self.links = pd.read_table(links_file)
        seqs = list(SeqIO.parse(fasta_file, format='fasta'))
        self.seqids = list(map(lambda x: x.id, seqs))
        self.seqdict = dict(zip(self.seqids, seqs))

    def random_peptide(self):
        i = np.random.randint(0, len(self.seqids))
        id_ = self.seqids[i]
        return str(self.seqdict[id_].seq)

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
        it = 0
        while True:
            try:
                line = self.handle.readline().lstrip()
                if it % w == worker_id:
                    toks = line.split(' ')
                    geneid = self.cols.loc['protein1']
                    posid = self.cols.loc['protein2']
                    neg = self.random_peptide()
                    gene = self.seqdict[geneid].sequence
                    pos = self.seqdict[posid].sequence
                    yield gene, pos, neg
                it += 1
            except StopIteration:
                self.handle = open(self.string_file, 'r')
                line = self.handle.readline().lstrip()
