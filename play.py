import torch as t
from bb import RV, Model, VI, HMC, Metropolis, Chain
from rdists import RNormal

def between(x, a, b):
    return (a < x) and (x < b)

class DifferentScales(Model):
    def __init__(self):
        super().__init__()
        self.a = RNormal((2,), loc=t.zeros(()), scale=t.ones(()))

    def __call__(self):
        scale = t.Tensor([1., 0.1])
        return t.distributions.Normal(self.a(), scale).log_prob(t.zeros(2))#.sum()

m = DifferentScales()

vi = VI(m)
vi.fit(3*10**4)

kernel = HMC(m.rvs(), ind_shape=[2])
chain = Chain(m, [kernel])
result = chain.run(1000)

#kernel = Metropolis(m.rvs(), vi=vi)
#chain = Chain(m, [kernel])
#result = chain.run(100000, warmup=10000)
