import unittest

import os
import random
import numpy as np
from poplar.dataset.genome_dataset import GenomeDataset
from poplar.dataset.data_util import distance
from poplar.util import get_data_path


class TestGenomeDataset(unittest.TestCase):
    def setUp(self):
        self.lam = get_data_path('lambda.gb')
        self.ecoli = get_data_path('ecoli.gb')

    def test_get_item(self):
        genes = GenomeDataset.read_genbank(self.ecoli)
        dataset = GenomeDataset([self.ecoli])
        res = dataset[0]
        self.assertEqual(len(res), 3)
        d = distance(res[0], res[1])
        self.assertLess(d, 10000)

    def test_random_gene(self):
        np.random.seed(0)
        genes = GenomeDataset.read_genbank(self.ecoli)
        dataset = GenomeDataset([self.ecoli])
        res = dataset.random_gene(genes)
        self.assertEqual(len(res), 3)
        d = distance(res[0], res[1])
        self.assertLess(d, 10000)

    def test_read_genbank(self):
        genes = GenomeDataset.read_genbank(self.ecoli)
        self.assertEqual(len(genes), 4321)
        self.assertEqual(genes[0].start, 189)
        self.assertEqual(genes[0].end, 255)
        self.assertEqual(genes[-1].start, 4638964)
        self.assertEqual(genes[-1].end, 4639651)


if __name__ == '__main__':
    unittest.main()
