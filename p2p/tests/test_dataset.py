import unittest
from p2p.util import get_data_path
import pandas as pd
from Bio import SeqIO
from p2p.dataset import InteractionDataset


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')

    def test_constructor(self):
        intsd = InteractionDataset(self.fasta_file, self.links_file)
        self.assertEqual(len(intsd.links), 99)
        self.assertEqual(len(intsd.seqs), 100)


if __name__ == "__main__":
    unittest.main()
