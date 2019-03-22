from bb import RV, Model, VI, HMC
import torch as t

from distributions import Normal, LogGamma, Gamma, Exponential

class Likelihood(Model):
    """
    """
    def __init__(self):
        super().__init__()
        self.N = 5
        self.T = 1000
        self.mean = Normal((self.N, 1), loc=0., scale=1.)
        self.scale = Gamma((self.N, 1), shape=1., scale=1.)

    def forward(self):
        return t.distributions.Normal(self.mean(), self.scale().expand(-1, self.T))

class Joint(Model):
    """

    """
    def __init__(self, likelihood, obs):
        super().__init__()
        self.likelihood = likelihood
        self.obs = obs

    def forward(self):
        return self.likelihood().log_prob(obs).sum()


like = Likelihood()
obs = like().sample()
true_latents = like.dump()

m = Joint(like, obs)
print(m())
m.refresh()
vi = VI(m)
vi.fit(10**4)
inferred_latents = like.dump()
