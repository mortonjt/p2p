import os
import unittest
from p2p.train import run, train
from p2p.util import get_data_path
import shutil

# import numpy as np
# from p2p.util import get_data_path
# import pandas as pd
# from Bio import SeqIO
# from p2p.dataset import InteractionDataset


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.links_file = get_data_path('links.txt')
        self.fasta_file = get_data_path('prots.fa')
        self.logging = 'logging'
        self.modelpath = 'model.pkt'

        # not ideal :(
        self.checkpoint = '/simons/scratch/jmorton/mgt/checkpoints/uniref50'
        self.data_dir = '/simons/scratch/jmorton/mgt/data/uniref50'

    def tearDown(self):
        # shutil.rmtree(self.logging, ignore_errors=True)
        os.remove(self.modelpath)

    # @unittest.skip("Run only in the presence of model or data")
    def test_run(self):
        acc1 = run(
            self.fasta_file, self.links_file,
            self.checkpoint, self.data_dir,
            self.modelpath, self.logging,
            training_column='Training',
            emb_dimension=10, num_neg=2,
            epochs=5, betas=(0.9, 0.95),
            batch_size=3, num_workers=0,
            summary_interval=1,
            device='cpu')
        os.path.exists(self.modelpath)


if __name__ == "__main__":
    unittest.main()
