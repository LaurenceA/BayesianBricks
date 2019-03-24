import torch as t
import torch.nn as nn
import torch.nn.functional as F

from rv import RV, Model

def sum_all_except(x, ind_dims):
    for dim in ind_dims:
        assert 1 < x.size(dim)

    sum_dims = set(range(len(x.size()))) - set(ind_dims)
    return x.sum(dim=tuple(sum_dims), keepdim=True)

class MCMC():
    """
    Assumes that the ratio of proposals forward and back is equal,
    which is true for symmetric proposals, and HMC.

    Critical technical detail:
    the ind_dims, and log_like must be compatible with all log_priors.
    """
    def __init__(self, rvs, ind_dims=(), vi=None):
        self.rvs = list(rvs)
        for rv in self.rvs:
            assert isinstance(rv, RV)
        self.ind_dims = ind_dims

        self.init_proposal(vi)

    def init_proposal(self, vi=None):
        raise NotImplementedError()

    def proposal(self):
        raise NotImplementedError()

    def log_prior(self):
        total = 0.
        for rv in self.rvs:
            total += sum_all_except(rv.log_prior(), self.ind_dims)
        return total


    def step(self, model):
        xs_prev = [rv._value.clone() for rv in self.rvs]
        lp_prev = model() + self.log_prior()

        self.proposal()

        xs_next = [rv._value.clone() for rv in self.rvs]
        lp_next = model() + self.log_prior()
        
        lp_diff = sum_all_except(lp_next - lp_prev, self.ind_dims)
        accept_prob = lp_diff.exp()
        accept_cond = t.rand(accept_prob.size()) < accept_prob

        for i in range(len(self.rvs)):
            new_value = t.where(
                accept_cond,
                xs_next[i],
                xs_prev[i]
            )
            assert new_value.size() == self.rvs[i].size
            self.rvs[i]._value = new_value


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
