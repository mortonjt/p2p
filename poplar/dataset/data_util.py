import numpy as np
import collections
from copy import copy


GeneInterval = collections.namedtuple(
    'GeneInterval', ['start', 'end', 'sequence', 'strand']
)

def get_seq(x):
    if 'translation' in x.qualifiers:
        return x.qualifiers['translation'][0]
    else:
        return ''

def get_context(genes, idx, window_size):
    """ Retrieves context genes

    Parameters
    ----------
    genes : list of GeneInterval
        List of genes in genome
    idx : int
        Index for gene of interest
    """
    lidx, ridx = idx - 1, idx + 1
    context = []

    # only grab gene if it is in the same strand
    while lidx >= 0 and ((genes[idx].start - genes[lidx].end) < window_size):
        if genes[lidx].strand == genes[idx].strand:
            context.append(genes[lidx])
        lidx = lidx - 1
    while ridx < len(genes) and ((genes[ridx].start - genes[idx].end) < window_size):
        if genes[ridx].strand == genes[idx].strand:
            context.append(genes[ridx])
        ridx = ridx + 1
    return context

def draw_exclusive(n, idx):
    if n <= 1:
        raise ValueError('Cannot draw exclusively, n<=1')
    j = np.random.randint(n)
    while j == idx:
        j = np.random.randint(n)
    return j

def overlap(x, y):
    """ Tests for overlap

    Parameters
    ----------
    x : GeneInterval
       First gene
    y : GeneInterval
       Second gene

    Returns
    -------
    True if overlap.  False otherwise
    """
    return x.start <= y.end and y.start <= x.end

def distance(x, y):
    """ Computes minimum distance between genes

    Parameters
    ----------
    x : GeneInterval
       First gene
    y : GeneInterval
       Second gene

    Returns
    -------
    int : Distance between genes
    """
    if overlap(x, y):
        return 0
    else:
        return min(
            abs(x.end - y.start),
            abs(y.end - x.start)
        )

def mask(x, prob=0.5, mask_chr='_'):
    y = copy(x)
    r = np.random.random(len(y))
    y[r<prob] = mask_chr
    return y

def mutate(x, prob, vocab):
    y = copy(x)
    r = np.random.random(len(y))
    y[r<prob] = vocab.random(np.sum(r<prob))
    return y

