import torch
import pyro
import pyro.infer
import pyro.poutine
from pyro.infer.mcmc import NUTS, HMC, MCMC
from pyro.ops.stats import effective_sample_size
import pyro.optim
import pyro.distributions as dist
from torch.autograd.profiler import profile


from timeit import default_timer as timer


pyro.set_rng_seed(1)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(torch.ones(()).device)

N_inputs = 100
N_outputs = 1000#
N_points = 8000

W_true = torch.randn(N_outputs, N_inputs)
lr_out_true = torch.randn(N_outputs, 1)
lr_in_true = torch.randn(1, N_inputs)

X = torch.randn(N_inputs, N_points)
y = dist.Normal(W_true @ X, 1.).sample()


def model(X):
    coefs_mean = torch.zeros(N_outputs, N_inputs)
    coefs_std = torch.ones(N_outputs, N_inputs)
    W = pyro.sample('W', dist.Normal(coefs_mean, coefs_std))
    lr_in  = pyro.sample('lr_in', dist.Normal(torch.zeros(1, N_inputs), 1.))
    lr_out = pyro.sample('lr_out', dist.Normal(torch.zeros(N_outputs, 1), 1.))
    pyro.sample('y', dist.Normal((W + lr_out@lr_in) @ X, 1.), obs=y)
    return W

#trace_kernel = NUTS(model, adapt_step_size=True, adapt_mass_matrix=False)
trace_kernel = NUTS(model, adapt_step_size=False, step_size=2E-3, adapt_mass_matrix=False)

start = timer()
mcmc_run = MCMC(trace_kernel, num_samples=300, warmup_steps=500).run(X)
torch.cuda.synchronize()
end = timer()
print(end-start)

marginal = mcmc_run.marginal(["W"])

W_mean = marginal.empirical["W"].mean
import numpy as np
print(np.corrcoef(W_true.cpu().numpy().ravel(), W_mean.cpu().numpy().ravel()))
