import numpy as np
import pandas as pd
import torch


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
       Pytorch dataloader.

    Returns
    -------
    float : average AUC

    TODO
    ----
    Make sure that the test/validation dataset are
    sorted by (1) taxonomy then by (2) protein1.
    """
    pass

def pairwise_auc(model, dataloader):
    """ Pairwise AUC comparison

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    float : average AUC
    """
    pass


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

