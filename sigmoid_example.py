from bb import RV, Model, VI, HMC
import torch as t

from distributions import Normal, LogGamma, Gamma, Exponential

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
        return self.likelihood().log_prob(obs).sum()


contrast = t.randn(10, 1000)
like = Likelihood(contrast)
obs = like().sample()
true_latents = like.dump()

m = Joint(like, obs)
print(m())
m.refresh()
vi = VI(m)
vi.fit(10**5)
inferred_latents = like.dump()

#hmc = HMC(m, 5000)
#hmc.run()
