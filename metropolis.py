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
            for vit in vi.vits:
                if rv is vit.rv:
                    self.proposal_scale = 0.3*vit.std()
                    break

        if not hasattr(self, "proposal_scale"):
            self.proposal_scale = 0.01


class Metropolis(MCMC):
    def init_proposal(self, vi):
        self.mcmc_tensors = [MetropolisTensor(rv, vi) for rv in self.rvs]

    def proposal(self, model):
        for mt in self.mcmc_tensors:
            mt.rv._value.add_(mt.proposal_scale*t.randn(mt.rv.size))
        return 0.
