from bb import RV, Model, VI, HMC
import torch as t

#from confidence import confidence, Thresholds

from distributions import Normal, LogGamma, Gamma, Exponential

class Likelihood(Model):
    """
    """
    def __init__(self, contrast_interval_1, contrast_interval_2, C):
        super().__init__()
        #All these are N x T
        #where N is the number of participants
        #and T is the number of trials
        self.contrast_interval_1 = contrast_interval_1
        self.contrast_interval_2 = contrast_interval_2

        self.C = C
        self.N = self.contrast_interval_1.size(0)
        self.T = self.contrast_interval_1.size(1)

        assert t.Size([self.N, self.T]) == self.contrast_interval_1.size()
        assert t.Size([self.N, self.T]) == self.contrast_interval_2.size()

        #self.thresholds = Thresholds(t.Size([self.N, 1]), C)

        #self.scale = Gamma((self.N, 1), shape=2., scale=0.1)
        self.scale = Exponential((self.N, 1), rate=5.)
        contrast_diff = contrast_interval_2 - contrast_interval_1
        self.observation = Normal([self.N, self.T], loc=contrast_diff, scale=self.scale)

        #self.noise_interval_1 = Normal([self.N, self.T], loc=0., scale=self.scale)
        #self.noise_interval_2 = Normal([self.N, self.T], loc=0., scale=self.scale)

    def forward(self):
        #observed_contrast_1 = self.contrast_interval_1 + self.noise_interval_1()
        #observed_contrast_2 = self.contrast_interval_2 + self.noise_interval_2()
        #z = observed_contrast_2 - observed_contrast_1
        #return t.distributions.Bernoulli(logits=self.observation())
        return t.distributions.Normal(self.observation(), 1.)

class Joint(Model):
    """

    """
    def __init__(self, likelihood, obs):
        super().__init__()
        self.likelihood = likelihood
        self.obs = obs

    def forward(self):
        return self.likelihood().log_prob(obs).sum()


N = 3
T = 1000

contrasts = t.Tensor([0.05, 0.07, 0.10, 0.15])
interval = t.randint(0, 2, (N, T)).float()
contrast_interval_1 = interval*contrasts[t.randint(0, len(contrasts), (N, T))]
contrast_interval_2 = (1-interval)*contrasts[t.randint(0, len(contrasts), (N, T))]

like = Likelihood(contrast_interval_1, contrast_interval_2, 8)
obs = like().sample()
true_latents = like.dump()

m = Joint(like, obs)
print(m())
m.refresh()
vi = VI(m)
vi.fit(10**4)
inferred_latents = like.dump()
