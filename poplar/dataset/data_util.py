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

    Note
    ----
    By the definition, genes are in the same operon if they can be
    transcribed under the same transcription factor.

    Since strands are transcribed separately, genes are different strands
    cannot be on the same operon.

    Another note, if two genes overlap, then they cannot be in the same operon
    for the same reason -- they cannot be transcribed under the same
    transcription factor.
    """
    lidx, ridx = idx - 1, idx + 1
    context = []

    # only grab gene if it is in the same strand
    while lidx >= 0 and ((genes[idx].start - genes[lidx].end) < window_size):
        if (genes[lidx].strand == genes[idx].strand and
            not overlap(genes[lidx], genes[idx])):
            context.append(genes[lidx])
        lidx = lidx - 1
    while ridx < len(genes) and ((genes[ridx].start - genes[idx].end) < window_size):
        if (genes[ridx].strand == genes[idx].strand and
            not overlap(genes[ridx], genes[idx])):
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
