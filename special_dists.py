import torch as t
import torch.nn.functional as F
#from numpy import prod

    

def diffusion(z, tau, inp=0., dt=1.):
    """
    Gaussian diffusion without decay (i.e. diffusion without drift). 
    Zero at time t=-1 (omitted), so one time-step of random noise is visible.
    """
    inp = (dt/tau) * inp
    z = t.sqrt(2*dt/tau) * z
    total = inp + z
    return t.cumsum(total)

def causal_conv(x, filt):
    """
    assumes inputs and filters are both 1d
    broadcasts nicely
    may be inefficient as it doesn't use minibatches where it could
    """

    x_size = x.size()
    filt = filt.expand(x_size[:-1]+t.Size([-1]))

    x = F.pad(x, (filt.size(-1), 0))

    x = x.view(1, -1, x.size(-1))
    filt = filt.view(-1, 1, filt.size(-1))
    assert x.size(1) == filt.size(0)
    return F.conv1d(x, filt, groups=x.size(1))[..., :-1].view(x_size)

#def exponential_filter(


def drift_diffusion(z, tau, inp=0., dt=1., init=0.):

    """
    in steady-state gives temporally correlated noise with variance 1

    init:
      ranges between 0 and 1, and describes how close we are to steady-state
      tau.size == [..., 1]
    """

    #doesn't do initial condition!

    dx = (dt/tau) * inp + t.sqrt(t.ones(())*2*dt/tau) * z
    filter_length = dx.size(-1)
    filt = t.exp(dt / tau * t.arange(-(z.size(-1)-1), 1.))

    return 


def lds(z, inp, init, tau, noise_matrix, input_matrix):
    """
    z, inp [..., channels, time]
    Arbitrary (inc. non-normal) dynamical systems:
     Latents undergo exponential decay (rotation?)
     Noise and inputs are projected arbitrarily onto latents.
    """


