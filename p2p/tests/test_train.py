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
        self.fasta_file = get_data_path('prots.fa')
        self.links_file = os.path.abspath('data/links_files')
        self.logging = 'logging'
        self.modelpath = 'model.pkt'

        # not ideal :(
        self.checkpoint = '/simons/scratch/jmorton/mgt/checkpoints/uniref50'
        self.data_dir = '/simons/scratch/jmorton/mgt/data/uniref50'

    def tearDown(self):
        # shutil.rmtree(self.logging, ignore_errors=True)
        # os.remove(self.modelpath)
        pass

    # @unittest.skip("Run only in the presence of model or data")
    def test_run(self):
        
        acc1 = run(
            self.fasta_file, self.links_file,
            self.checkpoint, self.data_dir,
            self.modelpath, self.logging,
            training_column=2,
            emb_dimension=100, num_neg=2,
            epochs=1, betas=(0.9, 0.95),
            batch_size=4, num_workers=4,
            summary_interval=100000,  
            checkpoint_interval=100000,
            device='cuda:0')

        os.path.exists(self.modelpath)


if __name__ == "__main__":
    unittest.main()
