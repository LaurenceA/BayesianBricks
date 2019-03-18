import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions import Normal
from collections import OrderedDict

from vi import VI
from hmc import HMC



class LDS():
    def __init__(self, channels, T):


def lds(z, init_z, taus, noise_matrix):
    """
    z, inp [..., channels, time]
    Arbitrary (inc. non-normal) dynamical systems:
     Latents undergo exponential decay (rotation?)
     Noise and inputs are projected arbitrarily onto latents.
    """


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


def fn(d):
    mean = torch.stack([d["a"], d["b"]])
    scale = t.Tensor([1., 0.01])
    return Normal(mean, scale)

lp = LP(fn, t.zeros(2))


vi = VI(lp, sd)#, batch_size=t.Size([20]))
vi.fit(10**4)
hmc = HMC(lp, vi, 1000, rate=1E-1, trajectory_length=1.)
hmc.run()

print(hmc.tensors.hmcts["a"].samples.var().item())
print(hmc.tensors.hmcts["b"].samples.var().item())

