import torch as t
from bb import RV, Model, VI, HMC, MHMC, Chain
from distributions import Normal

def between(x, a, b):
    return (a < x) and (x < b)

class DifferentScales(Model):
    def __init__(self):
        super().__init__()
        self.a = Normal((), loc=t.zeros(()), scale=t.ones(()))
        self.b = Normal((), loc=t.zeros(()), scale=t.ones(()))

    def __call__(self):
        a = self.a()
        b = self.b()
        mean = t.stack([a, b])
        scale = t.Tensor([1., 0.1])
        return t.distributions.Normal(mean, scale).log_prob(t.zeros(2)).sum()

m = DifferentScales()

vi = VI(m)
vi.fit(3*10**4)

kernel = MHMC(m.rvs())
chain = Chain(m, [kernel], 100000, warmup=10000)
chain.run()
