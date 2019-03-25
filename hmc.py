import torch as t
import torch.nn as nn
import torch.nn.functional as F

from mcmc import MCMC

class HMCTensor():
    def __init__(self, rv, vi):
        self.rv = rv
        self.p = t.zeros(rv.size)

        if vi is not None:
            self.inv_mass = vi[rv].variance()
            self.sqrt_mass = vi[rv].inv_std()
        else:
            self.inv_mass = t.ones(())
            self.sqrt_mass = t.ones(())

    def momentum_step(self, rate):
        self.p.add_( rate, self.rv._value.grad)

    def position_step(self, rate):
        self.rv._value.data.addcmul_(rate, self.inv_mass, self.p)

    def zero_grad(self):
        if self.rv._value.grad is not None:
            self.rv._value.grad.fill_(0.)

    def step_initialize(self):
        self.p.normal_(0., 1.)
        self.p.mul_(self.sqrt_mass)
        self.rv._value.requires_grad_()


class HMC(MCMC):
    def init_proposal(self, vi, rate=1E-2, trajectory_length=1.):
        self.rate = rate
        self.steps = int(trajectory_length // rate)
        self.mcmc_tensors = {rv: HMCTensor(rv, vi) for rv in self.rvs}

    def position_step(self, rate):
        for mt in self.mcmc_tensors.values():
            mt.position_step(rate)

    def momentum_step(self, rate):
        self.zero_grad()
        lp = self.log_prob()
        lp.backward()
        for mt in self.mcmc_tensors.values():
            mt.momentum_step(rate)

    def zero_grad(self):
        for mt in self.mcmc_tensors.values():
            mt.zero_grad()

    def step_initialize(self):
        for mt in self.mcmc_tensors.values():
            mt.step_initialize()

    def log_prior_p(self):
        total = 0.
        for mt in self.mcmc_tensors.values():
            total += self.sum_non_ind(-0.5*(mt.inv_mass*mt.p**2))
        return total

    def proposal(self):
        self.step_initialize()

        lp_mod_init = self.log_prior_p()
        self.momentum_step(0.5*self.rate)
        
        # Integration
        for _ in range(self.steps-1):
            self.position_step(self.rate)
            self.momentum_step(self.rate)
        self.position_step(self.rate)

        self.momentum_step(0.5*self.rate)
        lp_mod_prop = self.log_prior_p()

        return lp_mod_prop - lp_mod_init

    def __getitem__(self, key):
        return self.mcmc_tensors[key]
