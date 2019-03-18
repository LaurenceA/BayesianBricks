import math
import torch as t
import torch.nn.functional as F
from bb import RV, Model
from torch.distributions import Poisson

def matmul_channels(signal, mat):
    return (signal.transpose(-1, -2) @ mat).transpose(-1, -2)

def causal_conv(x, filt):
    """
    assumes inputs and filters are both 1d
    broadcasts nicely
    may be inefficient as it doesn't use minibatches where it could
    """

    assert 2 == len(filt.size())
    assert filt.size(-2) == x.size(-2)

    x_size = x.size()

    x = F.pad(x, (filt.size(-1), 0))

    x = x.view(-1, x.size(-2), x.size(-1))
    filt = filt.view(filt.size(-2), 1, filt.size(-1))
    return F.conv1d(x, filt, groups=x_size[-2])[..., :-1].view(x_size)

def exponential_causal_conv(x, tau, dt=1.):
    filt = t.exp(dt / tau * t.arange(-(x.size(-1)-1), 1.))
    return causal_conv(x, filt)

class LDS(Model):
    """
    z, inp [..., channels, time]
    Arbitrary (inc. non-normal) dynamical systems:
     Latents undergo exponential decay (rotation?)
     Noise and inputs are projected arbitrarily onto latents.
    """
    def __init__(self, channels, T, batch_size=t.Size([]), dt=1.):
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.T = T

        self.dt = dt

        self.log_tau    = RV([channels, 1])
        self.noise_proj = RV([channels, channels])
        self.init       = RV([*batch_size, channels, 1])
        self.noise      = RV([*batch_size, channels, T])

    def forward(self):
        #tau = self.log_tau().exp()
        tau = 10*t.ones(self.channels, 1)
        A = self.noise_proj() / math.sqrt(self.channels)
        projected_noise = matmul_channels(self.noise(), A)
        noise = t.sqrt(2*self.dt / tau) * self.noise()
        return exponential_causal_conv(noise, tau, dt=self.dt)


class PoissonLDS(Model):
    def __init__(self, lds, N, nonlin=F.softplus):
        super().__init__()
        self.lds = lds
        self.W = RV((lds.channels, N))
        self.nonlin = nonlin

    def forward(self):
        rates = self.nonlin(matmul_channels(self.lds(), self.W()))
        return Poisson(rates)
        

lds = LDS(2, 100)
lds()

plds = PoissonLDS(lds, 10)
