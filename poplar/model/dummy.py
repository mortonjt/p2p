import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import math


class DummyModel(nn.Module):
    """ Dummy one-hot encoding model.

    See the tutorial below for more details.
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    The main purpose of this model is to show how to build a simple
    transformer model.  But also for testing.
    """
    def __init__(self, input_size, hidden_size):

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(input_size, hidden_size)

    def extract_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z
