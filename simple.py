# Initialize dictionary of Gaussian IID variables.
# Several possible set-ups:
#   Option 1: (simple initial implementation)
#     Function mapping Gaussian IID latents onto log_prob.
#   Option 2: (sample from generative)
#     Function mapping Gaussian IID latents onto observation distributions
#   Option 3: (sample from generative, and record moments of latents)
#     Function mapping Gaussian IID latents onto "real" latents
#     Function mapping "real" latents onto observation distributions
# 
# Adaptation == black-box VI with a factorised prior

# TODO
# Unify class for VI and HMC
# Allow for batch processing
# Alternate between VI and HMC?

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions import Normal
from collections import OrderedDict

def fill(val, size_dict):
    result_dict = {}
    for key, size in size_dict.items():
        sample_dict[k] = t.Tensor(size).fill_(val)
    return result_dict

def randn(size_dict):
    result_dict = {}
    for key, size in size_dict.items():
        result_dict[key] = t.randn(size)
    return result_dict

def prior_log_prob(sample_dict):
    N = Normal(0., 1.)
    total = 0.
    for k, v in size_dict.items():
        total += N.log_prob(v)
    return total

class VITensor(nn.Module):
    def __init__(self, size, batch_size=t.Size([])):
        super().__init__()
        self.size = size
        self.batch_size = batch_size
        self.full_size = batch_size + size
        self.mean = nn.Parameter(t.randn(size))
        self.log_prec = nn.Parameter(t.Tensor(size).fill_(8.))

    def log_variance(self):
        return F.logsigmoid(-self.log_prec) # log(1 / (1+log_prec.exp()))

    def variance(self):
        return F.sigmoid(-self.log_prec)

    def std(self):
        return ( 0.5*self.log_variance()).exp()

    def inv_std(self):
        return (-0.5*self.log_variance()).exp()

    def sample(self, batch_size=t.Size([])):
        z = t.randn(batch_size + self.size)
        return (self.mean + self.std() * z).detach()

    def rsample_kl(self):
        z = t.randn(self.full_size)
        log_var = self.log_variance()
        log_scale = 0.5*log_var
        scale = log_scale.exp()
        x = self.mean + scale * z

        logp = - 0.5 * (x**2).sum()
        logq = - 0.5 * (z**2 + log_scale).sum()
        return x, logq - logp
        

class VI(nn.Module):
    def __init__(self, fn, size_dict, batch_size=t.Size([]), opt=torch.optim.Adam, opt_kwargs={}):
        self.fn = fn

        super().__init__()
        for key, size in size_dict.items():
            setattr(self, key, VITensor(size, batch_size=batch_size))

        self.opt = opt(self.parameters(), **opt_kwargs)

    def rsample_kl(self):
        result_dict = {}
        total_kl = 0.
        for k, v in self._modules.items():
            result_dict[k], kl = v.rsample_kl()
            total_kl += kl
        return result_dict, total_kl

    def fit_one_step(self):
        self.zero_grad()
        rsample, kl = self.rsample_kl()
        elbo = self.fn(rsample) - kl
        loss = -elbo
        loss.backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for _ in range(T):
            elbo = self.fit_one_step()
            print(elbo)

class HMCTensor():
    def __init__ (self, vit, chain_length=10, batch_size=t.Size([])):
        self.size = vit.siz
        self.batch_size = batch_size
        self.full_size = batch_size + size

        self.inv_mass = v.variance()
        self.sqrt_mass = v.inv_std()

        #record of samples
        self.record = t.zeros(t.Size([chain_length]) + self.full_size())

        #state of Markov chain 
        self.x_mcmc = v.sample(batch_size)

        #state of leapfrog integrator
        self.x_int = self.x_init.clone()
        self.p_int = t.randn(v.full_size)

    def 

class HMC():
    def __init__(self, fn, vi, rate=1E-2, trajectory_length=1.):
        vi.zero_grad()

        self.x_init = OrderedDict()
        self.x = OrderedDict()
        self.p = OrderedDict()
        self.inv_mass = OrderedDict()
        for k, v in vi._modules:
            self.x_init[k] = v.mean.clone()
            self.x[k]      = v.mean.clone()
            self.p[k]      = t.randn(v.size)
            self.inv_mass  = 1.









sd = {
    "a" : t.Size([]),
    "b" : t.Size([])
}

sample_dict = randn(sd)

def lp(d):
    return Normal(0, 1).log_prob(d["a"]).sum() + Normal(0, 0.01).log_prob(d["b"]).sum()

vi = VI(lp, sd)#, batch_size=t.Size([20]))
