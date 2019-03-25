import torch as t

from rdists import RNormal, RExponential
from rv import Bernoulli, pt

from bb import VI
from bb import VI, Metropolis, Chain

N = 10
T = 1000
contrast = t.randn(10, 1000)

scale = RExponential((N, 1), rate=1.)
perceived_contrast = RNormal((N, T), loc=contrast, scale=scale)
logits = pt(lambda x: 10*x)(perceived_contrast)
choice = Bernoulli((N, T), logits=logits)

#data obtained by sampling from the model
data = choice()
true_latents = choice.dump()
choice.condition(data)
print(choice.log_prob().sum())

#reset all randomness in model
choice.sample()
print(choice.log_prob().sum())

# Initialize with VI
vi = VI(choice)
vi.fit(3*10**4)
print(choice.log_prob().sum())

# Finish up with MCMC
k1 = Metropolis(choice, [perceived_contrast.z], ind_shape=(10, 1000), vi=vi)
k2 = Metropolis(choice, [scale.z], ind_shape=(10, 1), vi=vi)
chain = Chain([k1, k2], 10000)

print(choice.log_prob().sum())
