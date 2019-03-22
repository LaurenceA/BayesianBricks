from bb import RV, Model, VI, HMC
import torch as t

from distributions import Normal, LogGamma, Gamma, Exponential
from confidence import confidence#, Threshold

t.set_default_tensor_type(t.cuda.FloatTensor)

class Likelihood(Model):
    """
    """
    def __init__(self, contrast):
        super().__init__()
        self.N = contrast.size(0)
        self.T = contrast.size(1)
        self.scale = Exponential((self.N, 1, 1), rate=1.)
        self.obs = Normal((self.N, self.T, 1), loc=contrast, scale=self.scale)
        #self.obs1 = Normal((self.N, self.T), loc= t.relu( contrast), scale=self.scale)
        #self.obs2 = Normal((self.N, self.T), loc=-t.relu(-contrast), scale=self.scale)

    def forward(self):
        return confidence(self.obs(), t.arange(-2, 3.), 5.) #t.distributions.Bernoulli(logits=10*(self.obs2 - self.obs1))

class Joint(Model):
    """

    """
    def __init__(self, likelihood, obs):
        super().__init__()
        self.likelihood = likelihood
        self.obs = obs

    def forward(self):
        return self.likelihood().log_prob(obs).sum()


contrast = t.randn(10, 1000, 1)
like = Likelihood(contrast)
obs = like().sample()
true_latents = like.dump()

m = Joint(like, obs)
print(m())
m.refresh()
vi = VI(m)
vi.fit(3*10**4)
inferred_latents = like.dump()

#hmc = HMC(m, 5000)
#hmc.run()
