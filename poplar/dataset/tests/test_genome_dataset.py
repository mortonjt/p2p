import unittest

import os
import random
import numpy as np
from poplar.dataset.genome_dataset import GenomeDataset
from poplar.util import get_data_path


class TestGenomeDataset(unittest.TestCase):
    def setUp(self):
        self.lam = get_data_path('lambda.gb')
        self.ecoli = get_data_path('ecoli.gb')

    def test_get_item(self):
        pass

    def test_get_item_cache(self):
        pass

    def test_iter(self):
        pass

    def test_random_gene(self):
        pass

    def test_read_genbank(self):
        genes = GenomeDataset.read_genbank(self.ecoli)
        self.assertEqual(len(genes), 4321)
        self.assertEqual(genes[0].start, 189)
        self.assertEqual(genes[0].end, 255)
        self.assertEqual(genes[-1].start, 4638964)
        self.assertEqual(genes[-1].end, 4639651)


if __name__ == '__main__':
    unittest.main()
