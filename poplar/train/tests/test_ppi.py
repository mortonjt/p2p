import os
import unittest
from poplar.train.ppi import ppi, train
from poplar.util import get_data_path, dictionary
from poplar.model.dummy import DummyModel
from poplar.model.ppibinder import PPIBinder
from poplar.dataset.interactions import NegativeSampler, InteractionDataDirectory

import shutil

# import numpy as np
# from poplar.util import get_data_path
# import pandas as pd
# from Bio import SeqIO
# from poplar.dataset import InteractionDataset


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.fasta_file = get_data_path('prots.fa')
        self.links_dir = os.path.abspath('data/links_files')
        # TODO:
        # 1. load dummy model
        # 2. fix simple_ppi with dummy model

        # Load dummy model
        input_dim = len(dictionary)
        hidden_size = 10
        self.emb_dimension = 3
        self.pretrained_model = DummyModel(input_dim, hidden_size)

        # freeze the weights of the pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.sampler = NegativeSampler(self.fasta_file)
        self.dataloader = InteractionDataDirectory(
            self.fasta_file, self.links_dir, training_column=4
        )
        self.pos_dataloader = [get_data_path('positives.txt')]
        self.neg_dataloader = [get_data_path('negatives.txt')]

        # setup model.
        self.ppi_model = PPIBinder(hidden_size, self.emb_dimension,
                                   self.pretrained_model)

    def test_train(self):
        max_steps = 1000
        learning_rate = 1e-3
        warmup_steps = 10
        gradient_accumulation_steps = 2
        clip_norm = 10
        summary_interval = 1
        checkpoint_interval = 360
        logging_path = 'logging_small_test'
        model_path = 'model.pkt'
        device = 'cpu'

        finetuned_model = train(
            self.ppi_model, self.dataloader,
            positive_dataloaders=self.pos_dataloader,
            negative_dataloaders=self.neg_dataloader,
            logging_path=logging_path,
            emb_dimension=self.emb_dimension,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            clip_norm=clip_norm,
            summary_interval=summary_interval,
            checkpoint_interval=checkpoint_interval,
            model_path=model_path,
            device=device
        )


class TestTrainingFull(unittest.TestCase):

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
        acc1 = ppi(
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
