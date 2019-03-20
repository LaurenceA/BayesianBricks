import math
import torch as t
from torch.distributions import Normal

#### Transformations of IID standard Gaussian noise
N = Normal(0., 1,)

def cdf(z):
    return N.cdf(z)

def ccdf(z):
    return N.cdf(-z)

def log_ccdf_direct(z):
    return N.cdf(-z).log()

def log_ccdf_expansion(z):
    # Only makes sense for large positive z
    # Based on the Mills Ratio (see Wikipedia)
    # Computes log( pdf(z)/z ( 1 - 1/z^2 + 3/z^4 - 3*5/z^6 + 3*5*7/z^8 ... ))
    log_prefac = N.log_prob(z) - t.log(z)
    zm2 = 1/z**2
    expansion = - zm2 * (1 - 3*zm2 * (1 - 5*zm2 * (1 - 7*zm2 * (1 - 9*zm2))))
    return log_prefac + t.log1p(expansion) 

def log_ccdf(z):
    return t.where(
        z < 4.5,
        log_ccdf_direct(z),
        log_ccdf_expansion(z)
    )

def log_cdf(z):
    return log_ccdf(-z)

    
#### Validating sizes of parameters and IID Gaussian noise



#### Functions to transform IID Gaussian noise into noise from a given distribution
def normal(z, loc, scale):
    assert t.all(t.zeros(()) <= scale)
    return loc + scale * z

def lognormal(z, mu, sigma):
    assert t.all(t.zeros(()) <= sigma)
    return (mu + sigma * z).exp()

def uniform(z, a, b):
    return a + (b-a)*cdf(z)

def exponential(z, rate):
    assert t.all(t.zeros(()) <= rate)
    return -log_ccdf(z)/rate

def laplace(z, loc, scale):
    assert t.all(t.zeros(()) <= scale)
    sl = t.where(
        z < 0,
           math.log(2) + log_cdf(z),
        - (math.log(2) + log_ccdf(z))
    )
    return loc + scale*sl

def gumbel(z, mu, scale):
    assert t.all(t.zeros(()) <= scale)
    return mu - scale*t.log(-log_cdf(z))

def logistic(z, loc, scale):
    assert t.all(t.zeros(()) <= scale)
    return loc + scale * (log_cdf(z) - log_ccdf(z))

def delta(z, loc):
    # 0*z ensures that the shape of the returned tensor is the same as for other distributions
    return 0*z+loc

def pareto(z, shape, scale):
    assert t.all(t.zeros(()) <= shape)
    assert t.all(t.zeros(()) <= scale)
    return scale * t.exp(-log_ccdf(z)/shape)

def gammaish(z, shape, scale):
    assert t.all(4.*t.ones(()) < shape)
    d = shape - 1/3
    v = (1+z*t.rsqrt(9*d))**3
    return scale*d*v

def trans(z, dist):
    """
    Allows us to use:
    Cauchy
    Half Cauchy
    Half Normal
    """
    return dist.icdf(cdf(z))

from bb import RV, Model
class AbstractNormal():
    dist = staticmethod(normal)
    
class AbstractLogNormal():
    dist = staticmethod(lognormal)

class AbstractUniform():
    dist = staticmethod(uniform)

class AbstractExponential():
    dist = staticmethod(exponential)

class AbstractLaplace():
    dist = staticmethod(laplace)

class AbstractGumbel():
    dist = staticmethod(gumbel)

class AbstractLogistic():
    dist = staticmethod(gumbel)

class AbstractDelta():
    dist = staticmethod(delta)

class AbstractPareto():
    dist = staticmethod(pareto)

class AbstractGammaish():
    dist = staticmethod(gammaish)


class FixedDist(Model):
    def __init__(self, size, *args):
        super().__init__()
        self.z = RV(size)
        for arg in args:
            assert not (isinstance(arg, RV) or isinstance(arg, Model))
        self.args = args

    def forward(self):
        return self.dist(self.z(), *self.args)

class FNormal(AbstractNormal, FixedDist): pass
class FLogNormal(AbstractLogNormal, FixedDist): pass
class FUniform(AbstractUniform, FixedDist): pass
class FExponential(AbstractExponential, FixedDist): pass
class FLaplace(AbstractLaplace, FixedDist): pass
class FGumbel(AbstractGumbel, FixedDist): pass
class FLogistic(AbstractLogistic, FixedDist): pass
class FDelta(AbstractDelta, FixedDist): pass
class FPareto(AbstractPareto, FixedDist): pass
class FGammaish(AbstractGammaish, FixedDist): pass



class HierarchicalDist(Model):
    def __init__(self, size, *args):
        super().__init__()
        self.z = RV(size)
        for arg in args:
            assert isinstance(arg, RV) or isinstance(arg, Model)
        self.args = args

    def forward(self):
        return self.dist(self.z(), *(arg() for arg in self.args))

class HNormal(AbstractNormal, HierarchicalDist): pass
class HLogNormal(AbstractLogNormal, HierarchicalDist): pass
class HUniform(AbstractUniform, HierarchicalDist): pass
class HExponential(AbstractExponential, HierarchicalDist): pass
class HLaplace(AbstractLaplace, HierarchicalDist): pass
class HGumbel(AbstractGumbel, HierarchicalDist): pass
class HLogistic(AbstractLogistic, HierarchicalDist): pass
class HDelta(AbstractDelta, HierarchicalDist): pass
class HPareto(AbstractPareto, HierarchicalDist): pass
class HGammaish(AbstractGammaish, HierarchicalDist): pass

#Others (with tractable cdf/icdfs):
#Generalised logistic

#### Gammaish


