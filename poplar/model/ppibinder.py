import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
from poplar.util import encode as encode_f
import math


class PPIBinder(nn.Module):
    def __init__(self, input_size, emb_dimension, peptide_model):
        """ Initialize model parameters.

        Parameters
        ----------
        input_size: int
            Input dimension size
        emb_dimention: int
            Embedding dimention, typically from 50 to 500.
        peptide_model : torch.nn.Module
            Language model for learning a representation of peptides.

        Notes
        -----
        The language_model must be a subclass of torch.nn.Module
        and must also have an `extract_features` method, which takes
        in as input a peptide encoding an outputs a latent representation.

        """
        # See here: https://adoni.github.io/2017/11/08/word2vec-pytorch/
        super(PPIBinder, self).__init__()
        self.input_size = input_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Linear(input_size, emb_dimension)
        self.v_embeddings = nn.Linear(input_size, emb_dimension)
        self.peptide_model = peptide_model
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.emb_dimension)
        self.u_embeddings.weight.data.normal_(0, initstd)
        self.v_embeddings.weight.data.normal_(0, initstd)

    def encode(self, x):
        f = lambda x: self.peptide_model.extract_features(encode_f(x))[:, 0, :]
        y = list(map(f, x))
        z = torch.cat(y, 0)
        return z

    def forward(self, pos_u, pos_v, neg_v):

        # only take <s> token for pos_u, pos_v, and neg_v
        # this will obtain prot embedding
        losses = 0
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, -1)
        score = F.logsigmoid(score)
        if score.dim() >= 1:
            losses += sum(score)
        else:
            losses += score
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v.unsqueeze(1),
                              emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        if neg_score.dim() >= 1:
            losses += sum(neg_score)
        else:
            losses += neg_score
        return -1 * losses

    def predict(self, x1, x2):
        emb_u = self.u_embeddings(x1)
        emb_v = self.v_embeddings(x2)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = F.logsigmoid(torch.sum(score, -1))
        return score
