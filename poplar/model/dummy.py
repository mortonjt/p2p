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
    def __init__(self, input_size, hidden_size, max_length=1024):
        super(DummyModel, self).__init__()
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(input_size, hidden_size)
        self.max_length = max_length

    def extract_features(self, x):
        y = self.encoder(x)
        if len(y.shape) == 2:
            y = y.view(1, y.shape[0], y.shape[-1])
        z = y.mean(1)
        z = z.view(y.shape[0], 1, y.shape[-1])
        u = torch.cat((z, y), 1)
        # pad with zeros
        if self.max_length > u.shape[1]:
            q = self.max_length - u.shape[1]
            zos = torch.zeros(u.shape[0], q, u.shape[2])
            u = torch.cat((u, zos), 1)
        return u

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z
