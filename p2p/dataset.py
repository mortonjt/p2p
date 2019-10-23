import numpy as np
from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd



class PeptideDataset(DataSet):

    def __init__(self, fasta_file, gene2acc, biogrid, num_neg=10):
        """ Read in fasta file

        Parameters
        ----------
        fasta_file : filepath
            Fasta file of sequences of interest.
        gene2acc : filepath
            Gene to accession mappings - subsetted to the sequences of interest.
        biogrid : filepath
            Table of interactions from biogrid.
        num_neg : int
            Number of negative samples.
        """
        fasta_seqs = SeqIO.parse(open(fasta_file, 'r'), 'fasta')
        self.seqids = list(map(lambda x: x.id))
        self.seqdict = dict(zip(seqids, list(fasta_seqs)))
        self.g2a = pd.read_table(gene2acc)
        self.interactions = pd.read_Table(biogrid)
        self.num_neg = num_neg
        self.g2a = self.g2a.set_index('GeneID')

    def __len__(self):
        return self.interactions.shape[0]

    def random_peptide(self):
        i = np.random.randint(0, len(self.seqids))
        id_ = seld.seqids[i]
        return self.seqdict[id_]

    def __getitem__(self, idx):
        """ Prepare triples for training (gene, pos, neg)"""
        row = self.interactions.loc[idx]
        col1 = 'Entrez Gene Interactor A'
        col2 = 'Entrez Gene Interactor B'
        gene_id = row[col1]
        pos_id = row[col2]
        gene = self.g2a.loc[gene_id, 'protein_accession.version'][0]
        pos = self.g2a.loc[pos_id, 'protein_accession.version'][0]
        neg = self.random_peptide()
        return gene, pos, neg

    def __iter__(self):
        """ Prepare triples for training (gene, pos, neg)"""
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.__len__())
        if worker_info is None:  # single-process data loading
            for i in range(end):
                for _ in self.num_neg:
                    gene, pos, neg = self.__getitem__(i)
                    yield gene, pos, neg
        else:
            # setup bounds
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                for _ in self.num_neg:
                    gene, pos, neg = self.__getitem__(i)
                    yield gene, pos, neg
