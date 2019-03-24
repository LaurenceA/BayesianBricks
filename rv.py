import torch as t
import torch.nn as nn
import torch.nn.functional as F

from vi import NormalVI

class AbstractRV(nn.Module):
    def vi(self):
        raise NotImplementedError()

    def sample(self):
        self._value = self.dist().sample()
        assert self.size == self._value.size()

    def log_prior(self):
        return self.dist().log_prob(self._value)

    def forward(self):
        return self._value

class BrRV(AbstractRV):
    """
    Never overload forward!  Can't take arguments (so as to encode)
    """
    def __init__(self, size, **kwargs):
        super().__init__()
        self.size = size
        self._kwargs = kwargs

        # for easy access in future, and to ensure that they are recorded as modules,
        # record keyword arguments as fields
        for k, v in _kwargs.items():
            setattr(self, k, v)

    def dist(self):
        return self._dist(**{k:unwrap(v) for k,v in self._kwargs.items()})

class LeafRV(AbstractRV):
    def dist(self):
        return self._dist

class NormalRV(AbstractRV):
    _dist = t.distributions.Normal

    def vi(self):
        return NormalVI(self)

class RV(NormalRV, LeafRV):
    def __init__(self, size):
        Model.__init__(self)
        self.size = t.Size(size)
        self._value = t.randn(size)
        self._dist = t.distributions.Normal(t.zeros(()).expand(self.size), t.ones(()))


class Model(nn.Module):
    """
    Overload: 
    __init__ 
    __call__
    """
    #def rvs(self):
    #    return (mod for mod in self.modules() if isinstance(mod, RV))
    
    def all_rvs(self):
        return (mod for mod in self.modules() if isinstance(mod, AbstractRV))

    def branch_rvs(self):
        return (mod for mod in self.modules() if isinstance(mod, BrRV))

    def leaf_rvs(self):
        return (mod for mod in self.modules() if isinstance(mod, LeaffRV))

    def all_named_rvs(self):
        return ((k, v) for k, v in self.named_modules() if isinstance(v, AbstractRV))

    def models(self):
        return (mod for mod in self.modules() if isinstance(mod, Model))

    def refresh(self):
        for rv in self.all_rvs():
            rv.sample()

    def dump(self):
        result = {}
        for k, v in self.named_modules():
            if hasattr(v, "_value"):
                result[k] = v._value
            else:
                result[k] = None
        return result
