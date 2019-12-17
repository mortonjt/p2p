import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import math


class RobertaClassificationHead(nn.Module):

    def __init__(self, input_dim, inner_dim, num_classes,
                 activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaConstrastiveHead(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        """ Initialize model parameters.

        Parameters
        ----------
        emb_size: int
            Embedding size.
        emb_dimention: int
            Embedding dimention, typically from 50 to 500.

        TODO:
          1. Rename this transformer file to something else (i.e. ppi).
          2. Rename RobertaConstrastiveHead to PPBinder
        """
        # See here: https://adoni.github.io/2017/11/08/word2vec-pytorch/
        super(RobertaConstrastiveHead, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        # TODO: swap u and v with linear layers
        self.u_embeddings = nn.Linear(emb_size, emb_dimension)
        self.v_embeddings = nn.Linear(emb_size, emb_dimension)
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.emb_dimension)
        self.u_embeddings.weight.data.normal_(0, initstd)
        self.v_embeddings.weight.data.normal_(0, initstd)

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

    def predict(self, pos_u, pos_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = F.logsigmoid(torch.sum(score, -1))
        return score
