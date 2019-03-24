from bb import RV, Model, VI
import torch as t

from rdists import RNormal, RLogGamma, RGamma, RExponential, RBeta
from confidence import confidence#, Threshold

#t.set_default_tensor_type(t.cuda.FloatTensor)

class Likelihood(Model):
    """
    """
    def __init__(self, contrast):
        super().__init__()
        self.N = contrast.size(0)
        self.T = contrast.size(1)
        self.scale = RExponential((self.N, 1, 1), rate=1.)
        self.obs1 = RNormal((self.N, self.T, 1), loc= t.relu( contrast), scale=self.scale)
        self.obs2 = RNormal((self.N, self.T, 1), loc=-t.relu(-contrast), scale=self.scale)
        self.beta = RBeta((self.N, 1, 1), alpha=0.5, beta=0.5)

    def forward(self):

        diff = self.obs2() - self.obs1()
        m1 = diff
        m2 = t.sign(diff) * t.max(self.obs2(), self.obs1())
        b = self.beta() 

        f = b * m1 + (1-b)*m2
       
        return confidence(f, t.arange(-2, 3.), 5.)

class Joint(Model):
    """

    """
    def __init__(self, likelihood, obs):
        super().__init__()
        self.likelihood = likelihood
        self.obs = obs

    def forward(self):
        return self.likelihood().log_prob(obs).sum()


contrast = t.randn(10, 5000, 1)
like = Likelihood(contrast)
obs = like().sample()
true_latents = like.dump()
m = Joint(like, obs)
m.refresh()

vi = VI(m)
vi.fit(3*10**4)
inferred_latents = like.dump()


