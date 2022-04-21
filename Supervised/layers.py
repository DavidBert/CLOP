import numpy as np
import torch.nn as nn
import math 
import torch
      
class CLOPLayer(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p
        
    def _shuffle(self, x):
        batch_size = x.shape[0]
        nb_channels = x.shape[1]
        flat = x.flatten(start_dim=2, end_dim=-1)
        idx = self._index_permute(x[0,0]).to(x.device)
        idx = idx.repeat(batch_size, nb_channels, 1)
        res = torch.gather(flat, 2, idx)
        return res.view_as(x)
    
    def _index_to_coord(self, index, nb_col):
        return (index // nb_col, index % nb_col)
    
    def _index_permute(self, x):
        n_element = x.nelement()
        nb_raw = x.shape[0]
        nb_col = x.shape[1]
        indexes = torch.arange(0, n_element, dtype=int).view_as(x)
        p = (1-self.p, self.p/4, self.p/4, self.p/4, self.p/4)
        for idx in torch.randperm(n_element):
            i, j = self._index_to_coord(idx, nb_raw)
            r = np.random.choice([0, 1, 2, 3, 4], p=p)
            if r != 0:
                if r == 1:
                    idx_prime = ((i + 1) % nb_raw, j)
                if r == 2:
                    idx_prime = ((i - 1) % nb_raw, j)
                if r == 3:
                    idx_prime = (i, (j + 1) % nb_col)
                if r == 4:
                    idx_prime = (i, (j - 1) % nb_col)
                tmp = int(indexes[i,j])
                indexes[i, j] = indexes[idx_prime]
                indexes[idx_prime] = tmp
        return indexes.flatten()
    
    def forward(self, x):
        if self.training:
            shuffled = self._shuffle(x)
            return shuffled
        return x 