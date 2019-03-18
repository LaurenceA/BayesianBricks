import torch as t
from vi import VI
from hmc import HMC

class MTensor():
    def __init__(self, size):
        self.value = t.randn(size)

    def __call__(self):
        return self.value

    def refresh(self):
        return t.randn(self.size)

class Normal(MTensor):
    def __call__(self, loc, scale):
        return loc + scale*self.value


class MDict():
    """
    define your own __init__
    define your own __call__
    """
    def __init__(self):
        self.tensors = {}

    def register_rv(self, name, rv):
        setattr(self, name, rv)
        self.tensors[name] = rv

    def refresh(self):
        for rv in self.tensors.values():
            rv.refresh()

        
class Model(MDict):
    def __init__(self):
        super().__init__()
        self.register_rv("a", Normal(()))
        self.register_rv("b", Normal(()))

    def __call__(self):
        a = self.a(0., 1.)
        b = self.b(0., 1.)
        mean = t.stack([a, b])
        scale = t.Tensor([1., 0.01])
        return t.distributions.Normal(mean, scale).log_prob(t.zeros(2)).sum()

m = Model()
lp = m()

VI(m)        

