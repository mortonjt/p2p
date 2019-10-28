import unittest
from p2p.util import get_data_path
import pandas as pd
from Bio import SeqIO


class TestInteractionDataset(unittest.TestCase):

    def setUp(self):
        links_file = get_data_path('links.txt')
        fasta_file = get_data_path('prots.fa')
        self.links = pd.DataFrame(links_file)
        seqs = list(SeqIO.parse(fasta_file))
        ids = list(map(lambda x: x.id, seqs))
        self.seqs = dict(zip(ids, seqs))

    def test_constructor(self):
        pass


if __name__ == "__main__":
    unittest.main()
