import numpy as np
import pandas as pd
import torch
from poplar.util import encode, tokenize


# Global evaluation metrics
# these are used mainly for testing evaluation
def mrr(model, dataloader):
    """ Mean reciprocial ranking.

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    float : mean reciprocial ranking
    """
    pass

def roc_auc(model, dataloader, k=10):
    """ ROC AUC

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader for validation data

    Returns
    -------
    float : average AUC

    TODO
    ----
    Make sure that the test/validation dataset are
    sorted by (1) taxonomy then by (2) protein1.
    """

def pairwise_auc(pretrained_model, binding_model,
                 dataloader, name, it, writer,
                 device='cpu'):
    """ Pairwise AUC comparison

    Parameters
    ----------
    language_model : popular.model
       Language model.
    binding_model : popular.model
       Binding prediction model.
    dataloader : torch.DataLoader
       Pytorch dataloader.
    name : str
       Name of the database used in dataloader.
    it : int
       Iteration number.
    writer : SummaryWriter
       Tensorboard writer.
    device : str
       Device name to transfer model data to.

    Returns
    -------
    float : average AUC
    """
    with torch.no_grad():
        rank_counts = 0
        batch_size = dataloader.batch_size
        for j, (gene, pos, rnd, tax, protid) in enumerate(dataloader):
            gv, pv, nv = tokenize(gene, pos, rnd,
                                  pretrained_model, device)
            cv_score = binding_model.forward(gv, pv, nv)
            pred_pos = binding_model.predict(gv, pv)
            pred_neg = binding_model.predict(gv, nv)
            cv_err += cv_score.item()
            pos_score += torch.sum(pred_pos).item()
            rank_counts += torch.sum(pred_pos > pred_neg).item()

        total = max(1, len(dataloader))
        print('dataloader', len(dataloader))
        avg_rank = rank_counts / total
        tpr = avg_rank / batch_size
        print(f'rank_counts {avg_rank}, tpr {tpr}, iteration {it}')
        writer.add_scalar(f'{name}/pairwise/TPR', tpr, it)

    return tpr


# Taxon specific evaluation metrics
# these are used mainly for validation evaluation
def taxon_mrr(model, dataloader):
    """ Taxon specific Mean reciprocial ranking.

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    float : mean reciprocial ranking per taxon
    """
    pass

def taxon_roc_auc(model, dataloader, k=10):
    """ Taxon specific ROC AUC

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    np.array : average AUC per taxon

    TODO
    ----
    Make sure that the test/validation dataset are
    sorted by (1) taxonomy then by (2) protein1.
    """
    pass

def taxon_pairwise_auc(model, dataloader):
    """ Taxon specific pairwise AUC comparison

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    np.array : average AUC per taxon
    """
    pass
