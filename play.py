import torch as t

from rdists import RNormal, RExponential
from rv import Normal

from bb import VI, Metropolis, Chain

# Define the model
mean = RNormal((2,), loc=t.zeros(()), scale=t.ones(()))
obs = Normal((2,), loc=mean, scale=t.Tensor([1., 0.1]))

# Condition on data
obs.condition(t.zeros(2))

# Initialize with VI
vi = VI(obs)
vi.fit(3*10**4)

# Finish up with MCMC
kernel = Metropolis(obs, [mean.z], vi=vi)
chain = Chain(obs, [kernel], 100000)
