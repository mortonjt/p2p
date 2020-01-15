import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import glob
import collections
from poplar.dataset.data_util import draw_exclusive, get_context, get_seq, GeneInterval


class GenomeDataset(Dataset):
    def __init__(self, genbank_files, num_neg = 10, num_gene_samples=100,
                 window_size=10000):
        #glob.glob(genbank_directory, '*' + genbank_ext)
        self.genbank_files = genbank_files
        self.window_size = window_size
        self.num_neg = num_neg
        self.num_gene_samples = num_gene_samples
        self.idx = None
        self._genes = None

    def __len__(self):
        return len(self.genbank_files)

    def __getitem__(self, item):
        # check cache and only reload genes if it is another item
        if item != self.idx:
            gb_file = self.genbank_files[item]
            # return a list of items for each genome
            genes = GenomeDataset.read_genbank(gb_file)
            self.genes = genes
            self.idx = item

        # get random gene and another context gene
        return self.random_gene(self.genes)

    def __iter__(self):
        start = 0
        end = len(self)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            for i in range(end):
                for _ in range(self.num_gene_samples):
                    for _ in range(self.num_neg):
                        try:
                            yield self.__getitem__(i)
                        except:
                            print(self.genbank_files[i], 'could not be read.')
                            continue
        else:
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                for _ in range(self.num_gene_samples):
                    for _ in range(self.num_neg):
                        try:
                            yield self.__getitem__(i)
                        except:
                            print(self.genbank_files[i], 'could not be read.')
                            continue

    def random_gene(self, genes):
        """ Retrieve random gene and a pair

        Parameters
        ----------
        genes : list of GeneInterval
            List of genes

        Returns
        -------
        gen : GeneInterval
            Gene of interest
        ctx : GeneInterval
            Suggested context gene
        rand : GeneInterval
            Random gene

        Note
        ----
        All genes more than 1022 peptides are trimmed to 1022 peptides
        Question : Do we want to concatentate the <CLS> and terminal tokens?
                   For now, that answer is yes.
        """
        idx = np.random.randint(len(genes))
        operon = get_context(genes, idx, window_size=self.window_size)
        gen = genes[idx]
        jdx = np.random.randint(len(operon))
        ctx = operon[jdx]
        kdx = np.random.randint(len(genes))
        rand = genes[kdx]
        return gen, ctx, rand

    @staticmethod
    def read_genbank(gb_file):
        """ Read in genbank file and return a list of genes. """
        gb_record = SeqIO.read(open(gb_file, "r"), "genbank")
        cds = list(filter(lambda x: x.type == 'CDS', gb_record.features))
        starts = list(map(lambda x: int(x.location.start), cds))
        ends = list(map(lambda x: int(x.location.end), cds))
        strand = list(map(lambda x: x.strand, cds))
        seqs = list(map(get_seq, cds))
        res = zip(starts, ends, seqs, strand)

        # sequences with start, end and position
        res = list(filter(lambda x: len(x) > 0, res))
        res = list(map(lambda x: GeneInterval(
            start=x[0], end=x[1], sequence=x[2], strand=x[3]
        ), res))
        return res
