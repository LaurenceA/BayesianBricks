import torch as t
import torch.nn as nn
import torch.nn.functional as F

from rv import AbstractRV, Model

class MCMC():
    """
    Assumes that the ratio of proposals forward and back is equal,
    which is true for symmetric proposals, and HMC.

    Critical technical detail:
    the ind_dims, and log_like must be compatible with all log_priors.
    """
    def __init__(self, rvs=None, ind_shape=(), vi=None):
        self.rvs = list(rvs)
        for rv in self.rvs:
            assert isinstance(rv, AbstractRV)

        self.ind_shape = ind_shape
        self.ind_dims = [i for i in range(-len(ind_shape), 0) if 1 < ind_shape[i]] 

        self.init_proposal(vi)

    def init_proposal(self, vi):
        raise NotImplementedError()

    def proposal(self, model):
        raise NotImplementedError()

    def step(self, model):
        xs_prev = [rv._value.clone() for rv in self.rvs]
        lp_prev = self.log_prob(model)

        lp_mod = self.proposal(model)

        xs_next = [rv._value.clone() for rv in self.rvs]
        lp_next = self.log_prob(model)
        
        lp_diff = lp_next - lp_prev + lp_mod
        accept_prob = lp_diff.exp()
        accept_cond = t.rand(accept_prob.size()) < accept_prob

        for i in range(len(self.rvs)):
            new_value = t.where(
                accept_cond.detach(),
                xs_next[i].detach(),
                xs_prev[i].detach()
            )
            assert new_value.size() == self.rvs[i].size
            self.rvs[i]._value = new_value

    def log_prob(self, model):
        total = self.sum_non_ind(model())
        for rv in self.rvs:
            total = total + self.sum_non_ind(rv.log_prior())

        for br_rv in model.branch_rvs():
            if br_rv not in self.rvs:
                total = total + self.sum_non_ind(br_rv.log_prior())
        return total

    def sum_non_ind(self, x):
        for dim in self.ind_dims:
            assert self.ind_shape[dim] == x.size(dim)

        sum_dims = set(range(-len(x.size()), 0)) - set(self.ind_dims)
        result = x.sum(dim=tuple(sum_dims), keepdim=True)

        #### Remove singleton dimensions at the front
        while True:
            if (0 != len(result.size())) and (1 == result.size(0)):
                result.squeeze_(0)
            else:
                break
        return result



class Chain():
    """
    Manages the recording of MCMC samples
    """
    def __init__(self, model, kernels):
        self.model = model
        self.kernels = kernels

    def run(self, chain_length, warmup=0):
        #### warmup
        for i in range(warmup):
            for kernel in self.kernels:
                kernel.step(self.model)

        #### initialize result dict
        self.model() 
        result_dict = {}
        for k, m in self.model.named_modules():
            if hasattr(m, "_value"):
                result_dict[k] = t.zeros(t.Size([chain_length]) + m._value.size(), device="cpu")

        #### run chain
        for i in range(chain_length):
            for kernel in self.kernels:
                kernel.step(self.model)

            #Record current sample
            for k, m in self.model.named_modules():
                if hasattr(m, "_value"):
                    result_dict[k][i,...] = m._value

        return result_dict
