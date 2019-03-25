import torch as t
import torch.nn as nn
import torch.nn.functional as F

from vi import NormalVI

"""
Sampling must follow the shape of the DAG.
To avoid sampling the same variable twice when there is a diamond, we need to use 
"""

class Deps(nn.Module):
    """
    Behaviour that should be inherited by all classes that depend on random variables.
    """
    def named_self_modules(self):
        yield from self.named_modules()

    def rvs(self):
        return (v for k, v in self.named_rvs())

    def cond_rvs(self):
        return (v for k, v in self.named_cond_rvs())

    def uncond_rvs(self):
        return (v for k, v in self.named_uncond_rvs())

    def branch_rvs(self):
        return (v for k, v in self.named_branch_rvs())

    def leaf_rvs(self):
        return (v for k, v in self.named_leaf_rvs())

    def models(self):
        return (mod for mod in self.modules() if isinstance(mod, Model))


    def named_rvs(self):
        return ((k, v) for k, v in self.named_self_modules() if isinstance(v, AbstractRV))

    def named_cond_rvs(self):
        return ((k, v) for k, v in self.named_rvs() if v.is_conditioned)

    def named_uncond_rvs(self):
        return ((k, v) for k, v in self.named_rvs() if not v.is_conditioned)

    def named_branch_rvs(self):
        return ((k, v) for k, v in self.named_rvs() if isinstance(v, BrRV))

    def named_leaf_rvs(self):
        return ((k, v) for k, v in self.named_rvs() if isinstance(v, LeafRV))

    def named_models(self):
        return ((k, v) for k, v in self.named_self_modules() if isinstance(v, Model))

    def dump(self):
        result = {}
        for k, v in self.named_modules():
            if hasattr(v, "_value"):
                result[k] = v._value
            else:
                result[k] = None
        return result

class AbstractRV(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampled_bit = False
        self.is_conditioned = False

    def vi(self):
        raise NotImplementedError()

    def reset_sampled_bit(self):
        self.sampled_bit = False
        for mod in self.children():
            mod.reset_sampled_bit()

    def _sample(self):
        #### Sample has to follow the tree, if there are LeafRV's
        if not self.sampled_bit:
            for mod in self.children():
                mod._sample()
            self._value = self.dist().sample()
            assert self.size == self._value.size()
            self.sampled_bit = True

    def sample(self):
        self.reset_sampled_bit()
        self._sample()

    def log_prior(self):
        return self.dist().log_prob(self._value)

    def forward(self):
        return self._value

def unwrap(x):
    if isinstance(x, RV) or isinstance(x, Model):
        return x()
    if isinstance(x, float):
        return t.ones(())*x
    else:
        return x


class BrRV(AbstractRV, Deps):
    """
    Never overload forward!  Can't take arguments (so as to encode)
    """
    def __init__(self, size, **kwargs):
        super().__init__()
        self.size = size
        self.kwargs = kwargs
        self.is_conditioned = False

        # for easy access in future, and to ensure that they are recorded as modules,
        # record keyword arguments as fields
        for k, v in self.kwargs.items():
            setattr(self, k, v)

        self.sample()

    def dist(self):
        return self._dist(**{k:unwrap(v).expand(self.size) for k,v in self.kwargs.items()})

    def condition(self, obs=None):
        if obs is not None:
            assert self.size == obs.size()
            self._value = obs
        self.is_conditioned = True

    def dump(self):
        result = {}
        for k, v in self.named_modules():
            if hasattr(v, "_value"):
                result[k] = v._value
            else:
                result[k] = None
        return result

class LeafRV(AbstractRV):
    def dist(self):
        return self._dist

class Normal(BrRV):
    _dist = t.distributions.Normal
    def vi(self):
        return NormalVI(self)

class Bernoulli(BrRV):
    _dist = t.distributions.Bernoulli

class RV(LeafRV):
    def __init__(self, size):
        super().__init__()
        self.size = t.Size(size)
        self._value = t.randn(size)
        self._dist = t.distributions.Normal(t.zeros(()).expand(self.size), t.ones(()))
    def vi(self):
        return NormalVI(self)



class Model(Deps):
    def _sample(self):
        for mod in self.children():
            mod._sample()

    def reset_sampled_bit(self):
        for mod in self.children():
            mod.reset_sampled_bit()

#class Model(nn.Module):
#    """
#    Overload: 
#    __init__ 
#    __call__
#    """
#    #def rvs(self):
#    #    return (mod for mod in self.modules() if isinstance(mod, RV))
#    
#    def all_rvs(self):
#        return (mod for mod in self.modules() if isinstance(mod, AbstractRV))
#
#    def branch_rvs(self):
#        return (mod for mod in self.modules() if isinstance(mod, BrRV))
#
#    def leaf_rvs(self):
#        return (mod for mod in self.modules() if isinstance(mod, LeaffRV))
#
#    def all_named_rvs(self):
#        return ((k, v) for k, v in self.named_modules() if isinstance(v, AbstractRV))
#
#    def models(self):
#        return (mod for mod in self.modules() if isinstance(mod, Model))
#
#    def _sample(self):
#        for mod in self.children():
#            mod._sample()
#
#    def reset_sampled_bit(self):
#        for mod in self.children():
#            mod.reset_sampled_bit()
#
#    def dump(self):
#        result = {}
#        for k, v in self.named_modules():
#            if hasattr(v, "_value"):
#                result[k] = v._value
#            else:
#                result[k] = None
#        return result


def pt(fn):
    def inner(*args, **kwargs):
        return PointwiseTransform(fn, *args, **kwargs)
    return inner

class PointwiseTransform(Model):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self._args = args
        self._kwargs = kwargs

        for i in range(len(args)):
            setattr(self, "arg"+str(i), args[i])
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self):
        self._value = self.fn(
            *(unwrap(arg) for arg in self._args), 
            **{k:unwrap(v).expand(self.size) for k,v in self._kwargs.items()}
        )
        return self._value
