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
        self.logging1 = 'logging1'
        self.logging2 = 'logging2'
        self.modelpath = 'model.pkt'

        # not ideal :(
        # on popeye
        # self.checkpoint = '/simons/scratch/jmorton/mgt/checkpoints/uniref50'
        # self.data_dir = '/simons/scratch/jmorton/mgt/data/uniref50'
        # on rusty
        self.checkpoint = '/simons/scratch/jmorton/mgt/checkpoints/uniref50'
        self.data_dir = '/simons/scratch/jmorton/mgt/data/uniref50'
        self.checkpoint = '/mnt/home/jmorton/research/gert/data/full/uniref50/checkpoints'
        self.data_dir = '/mnt/home/jmorton/research/gert/data/full/uniref50/pretrain_data'

    def tearDown(self):
        # shutil.rmtree(self.logging, ignore_errors=True)
        # os.remove(self.modelpath)
        pass

    # @unittest.skip("Run only in the presence of model or data")
    def test_run(self):
        # # single gpu
        # question : why is accuracy not changing?
        # check -
        # 1. the variance of error estimate
        # 2. the gradients
        acc1 = run(
            self.fasta_file, self.links_file,
            self.checkpoint, self.data_dir,
            self.modelpath, self.logging1,
            training_column=2,
            emb_dimension=10, num_neg=10,
            max_steps=1000, learning_rate=5e-3,
            warmup_steps=0,
            batch_size=4, num_workers=4,
            summary_interval=1,
            checkpoint_interval=100000,
            device='cuda:0')

        # multiple gpus
        # acc1 = run(
        #     self.fasta_file, self.links_file,
        #     self.checkpoint, self.data_dir,
        #     self.modelpath, self.logging,
        #     training_column=2,
        #     emb_dimension=100, num_neg=2,
        #     epochs=1, learning_rate=5e-5,
        #     warmup_steps=0, gradient_accumulation_steps=1,
        #     fp16=False, batch_size=4, num_workers=12,
        #     summary_interval=60,
        #     checkpoint_interval=60,
        #     device='cuda')
        #
        # os.path.exists(self.modelpath)


if __name__ == "__main__":
    unittest.main()
