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

def pairwise_auc(model, dataloader, name, it, writer):
    """ Pairwise AUC comparison

    Parameters
    ----------
    model : popular.model
       Model to be evaluated.
    dataloader : torch.DataLoader
       Pytorch dataloader.
    name : str
       Name of the database used in dataloader.
    it : int
       Iteration number.
    writer : SummaryWriter
       Tensorboard writer.

    Returns
    -------
    float : average AUC
    """
    with torch.no_grad():
        rank_counts = 0
        batch_size = dataloader.batch_size
        for j, (gene, pos, rnd, tax, protid) in enumerate(test_dataloader):
            gv, pv, nv = tokenize(gene, pos, rnd,
                                  pretrained_model, device)
            cv_score = finetuned_model.forward(gv, pv, nv)
            pred_pos = finetuned_model.predict(gv, pv)
            pred_neg = finetuned_model.predict(gv, nv)
            cv_err += cv_score.item()
            pos_score += torch.sum(pred_pos).item()
            rank_counts += torch.sum(pred_pos > pred_neg).item()

        avg_rank = rank_counts / len(test_dataloader)
        tpr = avg_rank / batch_size
        print(f'rank_counts {avg_rank}, tpr {tpr}, iteration {it}')
        writer.add_scalar(f'{name}/pairwise/rank_count', cv_err, it)
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
