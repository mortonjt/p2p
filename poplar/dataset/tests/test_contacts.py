import unittest
import numpy as np
from poplar.util import get_data_path
import pandas as pd
from Bio import SeqIO
from poplar.dataset import ContactMapDataset


class TestContactMapDataset(unittest.TestCase):

    def setUp(self):
        self.directory = 'data'

    def test_constructor(self):
        ds = ContactMapDataset(self.directory)
        res_files = ['data/102L-A.npz', 'data/108L-A.npz',
                     'data/104L-A.npz', 'data/103L-A.npz',
                     'data/101M-A.npz', 'data/107L-A.npz',
                     'data/109L-A.npz']
        self.assertListEqual(ds.files, res_files)

    def test_len(self):
        ds = ContactMapDataset(self.directory)
        self.assertEqual(len(ds), 7)

    def test_getitem(self):
        exp_seq = ('MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF'
                   'DRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELK'
                   'PLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAM'
                   'NKALELFRKDIAAKYKELGYQG')
        ds = ContactMapDataset(self.directory)
        res_seq, res_cm = ds[4]
        self.assertEqual(exp_seq, res_seq)
        self.assertEqual((154, 154), res_cm.shape)

if __name__ == "__main__":
    unittest.main()
