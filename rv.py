import torch as t
import torch.nn as nn
import torch.nn.functional as F

class RV(nn.Module):
    """
    Call this directly
    """
    def __init__(self, size):
        super().__init__()
        self.size = t.Size(size)
        self._value = t.randn(size)

    def randn(self):
        self._value = t.randn(self.size)

    def __call__(self):
        return self._value

    def log_prior(self):
        return -0.5*(self._value**2)

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
