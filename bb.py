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
#   output distribution (#2)
#     fn : IID Gaussian noise -> observation distribution
#     class conditioned distributoin
#   nesting
#   confidence example
#   noisy integration example


import math
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
        self.mean = nn.Parameter(t.randn(size))
        self.log_prec = nn.Parameter(t.Tensor(size).fill_(8.))

    def log_variance(self):
        return F.logsigmoid(-self.log_prec) # log(1 / (1+log_prec.exp()))

    def variance(self):
        return t.sigmoid(-self.log_prec)

    def std(self):
        return ( 0.5*self.log_variance()).exp()

    def inv_std(self):
        return (-0.5*self.log_variance()).exp()

    def sample(self, batch_size=t.Size([])):
        z = t.randn(batch_size + self.size)
        return (self.mean + self.std() * z).detach()

    def rsample_kl(self, batch_size=t.Size([])):
        z = t.randn(batch_size + self.size)
        log_var = self.log_variance()

        scale = (0.5*log_var).exp()
        x = self.mean + scale * z

        logp = - 0.5 * (x**2).sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return x, logq - logp

class VIDict(nn.Module):
    def __init__(self, sizes):
        assert isinstance(sizes, dict)

        super().__init__()
        for key, val in sizes.items():
            if isinstance(val, dict):
                setattr(self, key, VIDict(val))
            else:
                setattr(self, key, VITensor(val))
                
    def sample(self, batch_size=t.Size([])):
        result_dict = {}
        for key, val in self._modules:
            result_dict[key] = val.sample(batch_size=batch_size)

    def rsample_kl(self):
        result_dict = {}
        total_kl = 0.
        for k, v in self._modules.items():
            result_dict[k], kl = v.rsample_kl()
            total_kl += kl
        return result_dict, total_kl
        

class VI():
    def __init__(self, fn, size_dict, batch_size=t.Size([]), opt=torch.optim.Adam, opt_kwargs={}):
        super().__init__()
        self.fn = fn
        self.tensors = VIDict(size_dict)
        self.opt = opt(self.tensors.parameters(), **opt_kwargs)

    def fit_one_step(self):
        self.tensors.zero_grad()
        rsample, kl = self.tensors.rsample_kl()
        elbo = self.fn(rsample) - kl
        loss = -elbo
        loss.backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for _ in range(T):
            elbo = self.fit_one_step()

class HMCTensor():
    def __init__ (self, vit, chain_length, batch_size=t.Size([])):
        self.size = vit.size
        self.batch_size = batch_size
        self.full_size = batch_size + self.size

        self.inv_mass = vit.variance()
        self.sqrt_mass = vit.inv_std()

        #record of samples
        self.samples = t.zeros(t.Size([chain_length]) + self.full_size, device="cpu")

        #state of Markov chain 
        self.x_mcmc = vit.sample(batch_size)

        #state of leapfrog integrator
        self.x = self.x_mcmc.clone().requires_grad_()
        self.p = t.zeros(vit.size)

        assert not self.x_mcmc.requires_grad
        assert     self.x.requires_grad
        assert not self.p.requires_grad

    def momentum_step(self, rate):
        self.p.add_(rate, self.x.grad)
        self.p.add_(-rate, self.x.data)

    def position_step(self, rate):
        self.x.data.addcmul_(rate, self.inv_mass, self.p)

    def zero_grad(self):
        if self.x.grad is not None:
            self.x.grad.fill_(0.)

    def log_prior_xp(self):
        lp_x = -0.5*(self.x**2).sum()
        lp_p = -0.5*(self.inv_mass*self.p**2).sum()
        return lp_x + lp_p

    def record_sample(self, i):
        self.samples[i,...] = self.x_mcmc
        

class HMC():
    def __init__(self, fn, vi, chain_length, warmup=0, rate=1E-2, trajectory_length=1.):
        self.fn = fn
        self.rate = rate
        self.steps = int(trajectory_length // rate)
        self.hmcts = {}
        self.chain_length = chain_length
        self.warmup = warmup

        for k, v in vi.tensors._modules.items():
            self.hmcts[k] = HMCTensor(v, self.chain_length)

    def __getitem__(self, key):
        return self.hmcts[key].x

    def zero_grad(self):
        for v in self.hmcts.values():
            v.zero_grad()

    def momentum_step(self, rate):
        self.zero_grad()
        lp = self.fn(self)
        lp.backward()
        for v in self.hmcts.values():
            v.momentum_step(rate)
        return lp

    def position_step(self, rate):
        for v in self.hmcts.values():
            v.position_step(rate)

    def log_prior_xp(self):
        total = 0.
        for v in self.hmcts.values():
            total += v.log_prior_xp()
        return total

    def step(self, i=None):
        # Randomise momentum
        for v in self.hmcts.values():
            v.p.normal_(0., 1.)
            v.p.mul_(v.sqrt_mass)

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

        acceptance_prob = (lp_prop - lp_init).exp()

        #Acceptance
        if t.rand(()) < acceptance_prob:
            for v in self.hmcts.values():
                v.x_mcmc.fill_(v.x)

        if i is not None:
            for v in self.hmcts.values():
                v.record_sample(i)

    def run(self):
        for _ in range(self.warmup):
            self.step()

        for i in range(self.chain_length):
            self.step(i)

class LP():
    """
    fields:
    fn
    obs

    fn takes IID random noise and returns the distribution under which obs is presumed to have been drawn

    importantly, an instance of LP looks like a function that takes iid random Gaussian noise (z), and returns a log-probability.
    """
    def __init__(self, fn, obs):
        self.fn = fn
        self.obs = obs

    def __call__(self, z):
        return self.fn(z).log_prob(self.obs).sum()


sd = {
    "a" : t.Size([]),
    "b" : t.Size([])
}

sample_dict = randn(sd)



#def lp(d):
#    return Normal(0, 1).log_prob(d["a"]).sum() + Normal(0, 0.01).log_prob(d["b"]).sum()

def fn(d):
    mean = torch.stack([d["a"], d["b"]])
    scale = t.Tensor([1., 0.01])
    return Normal(mean, scale)

lp = LP(fn, t.zeros(2))


vi = VI(lp, sd)#, batch_size=t.Size([20]))
vi.fit(10**4)
hmc = HMC(lp, vi, 1000, rate=1E-1, trajectory_length=1.)
hmc.run()

print(hmc.hmcts["a"].samples.var().item())
print(hmc.hmcts["b"].samples.var().item())

