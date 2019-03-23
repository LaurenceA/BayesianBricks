from bb import RV, Model, VI, HMC
import torch as t

from distributions import Normal, LogGamma, Gamma, Exponential, Beta
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
        self.obs1 = Normal((self.N, self.T, 1), loc= t.relu( contrast), scale=self.scale)
        self.obs2 = Normal((self.N, self.T, 1), loc=-t.relu(-contrast), scale=self.scale)
        self.beta = Beta((self.N, 1, 1), alpha=0.5, beta=0.5)

    def forward(self):

        diff = self.obs2() - self.obs1()
        m1 = diff
        m2 = t.sign(diff) * t.max(self.obs2(), self.obs1())
        b = self.beta() 

        f = b * m1 + (1-b)*m2
       
        return confidence(f, t.arange(-4, 5.)/2, 5.)

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

hmc = HMC(m, 100, rate=1E-2, trajectory_length=2.)
print(hmc.run())
