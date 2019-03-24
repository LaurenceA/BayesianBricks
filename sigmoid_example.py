from bb import RV, Model, VI, HMC
import torch as t

from rv import NonReparamNormal
from distributions import Normal, LogGamma, Gamma, Exponential
from bb import Metropolis, Chain

t.set_default_tensor_type(t.cuda.FloatTensor)

class Likelihood(Model):
    """
    """
    def __init__(self, contrast):
        super().__init__()
        self.N = contrast.size(0)
        self.T = contrast.size(1)
        self.scale = Exponential((self.N, 1), rate=1.)
        self.observation = Normal((self.N, self.T), loc=contrast, scale=self.scale)

    def forward(self):
        return t.distributions.Bernoulli(logits=10*self.observation())

class Joint(Model):
    """

    """
    def __init__(self, likelihood, obs):
        super().__init__()
        self.likelihood = likelihood
        self.obs = obs

    def forward(self):
        return self.likelihood().log_prob(obs)


contrast = t.randn(10, 1000)
like = Likelihood(contrast)
obs = like().sample()
true_latents = like.dump()

m = Joint(like, obs)
print(m().sum())
m.sample()
print(m().sum())
vi = VI(m)
vi.fit(3*10**4)
inferred_latents = like.dump()
print(m().sum())

#hmc = HMC(m, 5000)
#hmc.run()




k1 = Metropolis([like.observation.z], ind_shape=(10, 1000))#, vi=vi)
k2 = Metropolis([like.scale.z], ind_shape=(10, 1))#, vi=vi)
chain = Chain(m, [k1, k2])
result = chain.run(10000)
