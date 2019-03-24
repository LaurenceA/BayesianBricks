import torch as t
from bb import RV, Model, VI, HMC, Metropolis, Chain
from distributions import Normal

def between(x, a, b):
    return (a < x) and (x < b)

class DifferentScales(Model):
    def __init__(self):
        super().__init__()
        self.a = Normal((2,), loc=t.zeros(()), scale=t.ones(()))

    def __call__(self):
        scale = t.Tensor([1., 0.1])
        return t.distributions.Normal(self.a(), scale).log_prob(t.zeros(2))

m = DifferentScales()

vi = VI(m)
vi.fit(3*10**4)

kernel = HMC(m.all_rvs(), ind_shape=[2])
chain = Chain(m, [kernel])
result = chain.run(1000)

#kernel = Metropolis(m.all_rvs(), vi=vi)
#chain = Chain(m, [kernel])
#result = chain.run(100000, warmup=10000)
