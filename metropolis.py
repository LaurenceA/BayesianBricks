import torch as t
import torch.nn as nn
import torch.nn.functional as F

from vi import VI
from rv import RV
from mcmc import MCMC

class MetropolisTensor():
    def __init__(self, rv, vi):
        self.rv = rv
        if vi is not None:
            self.proposal_scale = 0.3*vi[rv].std()
        else:
            self.proposal_scale = 0.1

class Metropolis(MCMC):
    def init_proposal(self, vi):
        self.mcmc_tensors = {rv: MetropolisTensor(rv, vi) for rv in self.rvs}

    def proposal(self):
        for mt in self.mcmc_tensors.values():
            mt.rv._value.add_(mt.proposal_scale*t.randn(mt.rv.size))
        return 0.

    def __getitem__(self, key):
        return self.mcmc_tensors[key]
