import torch
import pyro
import pyro.infer
from pyro.infer.mcmc import NUTS, HMC, MCMC
from pyro.ops.stats import effective_sample_size
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(1)

dim = 75
true_coefs = torch.randn(dim)
data = torch.randn(74, dim)
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

def model(data):
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(dim)))
    bias  = pyro.sample('bias', dist.Normal(0., 1.))
    y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1) + bias), obs=labels)
    return y

nuts_kernel = NUTS(model, adapt_step_size=True)
mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=300).run(data)

marginal = mcmc_run.marginal(["bias", "beta"])

bias_mean = marginal.empirical["bias"].mean
bias_variance = marginal.empirical["bias"].variance
