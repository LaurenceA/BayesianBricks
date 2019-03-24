import torch as t
import torch.nn as nn
import torch.nn.functional as F

from vi import VI, VITensor
from rv import RV, Model
from mcmc import MCMC, Chain
from metropolis import Metropolis, MetropolisTensor


class HMCTensor():
    def __init__(self, rv, vi):
        self.rv = rv
        m.p = t.zeros(m.size)

        if vi is not None:
            for vit in vi.vits:
                if rv is vit.rv:
                    self.inv_mass = vit.variance()
                    self.sqrt_mass = vit.inv_std()
                    break

        if not hasattr(self, "inv_mass"):
            self.inv_mass = 1.
            self.sqrt_mass = 1.

    def momentum_step(self, rate):
        self.p.add_( rate, self.rv._value.grad)
        self.p.add_(-rate, self.rv._value.data)

    def position_step(self, rate):
        self.rv._value.data.addcmul_(rate, self.inv_mass, self.p)

    def zero_grad(self):
        if self.rv._value.grad is not None:
            self.rv._value.grad.fill_(0.)

    def log_prior_xp(self):
        lp_x = -0.5*(self.rv._value**2).sum()
        lp_p = -0.5*(self.inv_mass*self.p**2).sum()
        return lp_x + lp_p

    def step_initialize(self):
        self.p.normal_(0., 1.)
        self.p.mul_(self.sqrt_mass)
        self.rv._value.requires_grad_()


class HMC(MCMC):
    def init_proposal(self, vi, rate=1E-2, trajectory_length=1.):
        self.rate = rate
        self.steps = int(trajectory_length // rate)
        self.mcmc_tensors = [HMCTensor(rv, vi) for rv in self.rvs]

    def position_step(self, rate):
        for mt in self.mcmc_tensors:
            mt.position_step(rate)

    def momentum_step(self, rate):
        self.zero_grad()
        lp = self.model()
        lp.backward()
        for mt in self.mcmc_tensors():
            mt.momentum_step(rate)
        return lp

    def zero_grad(self):
        for mt in self.mcmc_tensors:
            mt.zero_grad()

    def step_initialize(self):
        for mt in self.mcmc_tensors:
            mt.hmc_step_initialize()

    def log_prior(self):
        total = 0.
        for mt in self.mcmc_tensors:
            total += mt.log_prior_xp()
        return total

    def proposal(self):
        self.step_initialize()

        lp_prior_xp = self.log_prior_xp()
        lp_like     = self.momentum_step(0.5*self.rate)
        lp_init     = lp_prior_xp + lp_like
        
        # Integration
        for _ in range(self.steps-1):
            self.position_step(self.rate)
            self.momentum_step(self.rate)
        self.position_step(self.rate)

        lp_like     = self.momentum_step(0.5*self.rate)
        lp_prior_xp = self.log_prior_xp()
        lp_prop     = lp_prior_xp + lp_like

        return lp_prop - lp_init
