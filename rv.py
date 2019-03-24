import torch as t
import torch.nn as nn
import torch.nn.functional as F

class AbstractRV(nn.Module):
    def __call__(self):
        return self._value

class RV(AbstractRV):
    """
    Call this directly
    """
    def __init__(self, size):
        super().__init__()
        self.size = t.Size(size)
        self._value = t.randn(size)

    def prior_sample(self):
        self._value = t.randn(self.size)

    def prior_log_prob(self):
        return -0.5*(self._value**2)

class NRRV(nn.Module):
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

    def prior_sample(self):
        self._value = self.dist().sample()
        assert self.size == self._value.size()

    def prior_log_prob(self):
        return self.dist().log_prob(self._value)

    def forward(self, **kwargs):
        if self.reparam:
            self._value   = self.dist(self.z._value, **self.kwargs())
        else:
            self.z._value = self.dist_(self._value, **self.kwargs())
        return self._value


class NormalRV(NNRV):
    _dist = t.distributions.Normal

    def vi(self):
        return NormalVI(self)



class Model(nn.Module):
    """
    Overload: 
    __init__ 
    __call__
    """
    def rvs(self):
        return (mod for mod in self.modules() if isinstance(mod, RV))

    def models(self):
        return (mod for mod in self.modules() if isinstance(mod, Model))

    def refresh(self):
        for rv in self.rvs():
            rv.randn()

    def dump(self):
        result = {}
        for k, v in self.named_modules():
            if hasattr(v, "_value"):
                result[k] = v._value
            else:
                result[k] = None
        return result
