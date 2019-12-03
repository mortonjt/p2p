import torch
import glob
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import pandas as pd


class ContactMapDataset(Dataset):

    def __init__(self, directory):
        self.directory = directory
        self.files = glob.glob(f'{directory}/*.npz')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        res = np.load(fname)
        return res['sequence'], res['A_ca_10A']

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        w = float(worker_info.num_workers)
        start = 0
        end = self.__len__()

        if worker_info is None:  # single-process data loading
            for i in range(end):
                yield self.__getitem__[i]
        else:
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                yield self.__getitem__[i]

