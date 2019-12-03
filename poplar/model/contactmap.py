import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable


class ContactMapLinear(nn.Module):
    """ Simple contact map prediction. """
    def __init__(self, input_dim, inner_dim):
        """

        Parameters
        ----------
        input_dim : int
            Number of input dimensions.
        inner_dim : int
            Number of embedding dimensions.
        """
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.P = Variable(torch.ones(input_dim, inner_dim))
        self.Q = Variable(torch.ones(inner_dim, input_dim))

    def forward(self, features, **kwargs):
        """ Predicts contact map.

        Note: make sure that the feature dimensions match.
        """
        x = features[:, 1:, :]
        res = torch.zeros(self.input_dim, self.input_dim)
        for i in range(self.input_dim):
            for j in range(i):
                u = x[:, i, :]
                v = torch.t(x[:, j, :])
                res[i, j] = u @ self.P @ self.Q @ v
        return res


class ContactMapConstastiveConv(nn.Module):
    """ Convolutional NN with negative sampling. """
    def __init__(self):
        pass

    def forward(self, features, **kwargs):
        pass
