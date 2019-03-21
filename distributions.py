import math
import torch as t
from torch.distributions import Normal as tNormal

#### Transformations of IID standard Gaussian noise
N = tNormal(0., 1,)
gamma_K = 5

def cdf(z):
    return N.cdf(z)

def ccdf(z):
    return N.cdf(-z)

def log_ccdf_direct(z):
    #Need to clamp z in order to avoid NaN gradients propagating back through t.where
    z = t.clamp(z, max=4.5)
    return N.cdf(-z).log()

def log_ccdf_expansion(z):
    # Only makes sense for large positive z
    # Based on the Mills Ratio (see Wikipedia)
    # Computes log( pdf(z)/z ( 1 - 1/z^2 + 3/z^4 - 3*5/z^6 + 3*5*7/z^8 ... ))
    z = t.clamp(z, min=4.5)
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



#### Functions to transform IID Gaussian noise into noise from a given distribution

def normal(z, loc, scale=None, log_scale=None, var=None, log_var=None, prec=None, log_prec=None):
    assert 1 == sum(x is not None for x in (scale, log_scale, var, log_var, prec, log_prec))

    if   scale is not None:
        pass
    elif log_scale is not None:
        scale = log_scale.exp()
    elif var is not None:
        scale = var.sqrt()
    elif log_var is not None:
        scale = (log_var/2).exp()
    elif prec is not None:
        scale = t.rsqrt(prec)
    elif log_prec is not None:
        scale = (-log_prec/2).exp()
    
    assert t.all(t.zeros(()) <= scale)


    return loc + scale * z

def lognormal(z, *args, **kwargs):
    return normal(z, *args, **kwargs).exp()

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

def loggamma(z, shape=None, scale=None, mean=None, log_scale=None):
    """
    Based on: A simple method for generating gamma variables, Marsaglia and Tsang, 2000 (ACM Transactions on Mathematical Software)

    This is a rejection-sampling based algorithm, with a rejection rate that becomes asymptotically close to 1 as k increases
    (0.992 at k=4 and 0.996 at k=8).

    These high acceptance rates justify dropping the rejection step, at least for reasonable k (maybe k > 4).
    
    To obtain lower values of the shape, k, we use another trick from the paper:
    G(alpha) = G(alpha+1) U^(1/alpha)

    The "basic" parameters are shape and log_scale.  Can derive log_scale from scale and shape from mean.
    """

    assert (mean is None) != (shape is None)
    assert (scale is None) != (log_scale is None)
    if scale is not None:
        assert t.all(t.zeros(()) <= scale)
        log_scale = t.log(scale)

    if mean is not None:
        shape = mean / log_scale.exp()
    assert t.all(t.zeros(()) <= shape)


    zG = z[0,...]
    zUs = z[1:,...]

    #K = zUs.shape[-1]
    assert zUs.size(0) == gamma_K

    _shape = shape + gamma_K

    # Sample Gamma with _shape = shape+k, and scale 1
    d = _shape - 1/3
    c = t.rsqrt(9*d)
    logG = t.log(d) + 3*t.log(1+c*zG)

    # Modify Gamma using uniforms
    logUs = log_cdf(zUs)
    return log_scale + logG + (logUs/(shape + t.arange(gamma_K, dtype=t.float).view(-1, *(1 for i in range(len(zUs.shape)-1))))).sum(0)
def loginvgamma(z, *args, **kwargs):
    return -loggamma(z, *args, **kwargs)
def invgamma(z, *args, **kwargs):
    return (-loggamma(z, *args, **kwargs)).exp()
def gamma(z, *args, **kwargs):
    return loggamma(z, *args, **kwargs).exp()

def logdirichlet(z, alphas):
    assert z.size(-1) == K+1
    assert z.size(-2) == alphas.size(-1)

    lg = loggamma(z, shape=alphas, scale=t.ones(()))
    return lg - t.logsumexp(lg, dim=-1)

def dirichlet(z, alphas):
    return logdirichlet(z, alphas).exp()

def beta(z, alpha, beta):
    assert z.size(0) == 2
    assert z.size(1) == gamma_K + 1
    x = gamma(z[0,...], alpha, t.ones(()))
    y = gamma(z[1,...], beta, t.ones(()))
    return x/(x+y)

def trans(z, dist):
    """
    Allows us to use:
    Cauchy
    Half Cauchy
    Half Normal
    """
    return dist.icdf(cdf(z))

from bb import RV, Model

def unwrap(x):
    if isinstance(x, RV) or isinstance(x, Model):
        return x()
    if isinstance(x, float):
        return t.ones(())*x
    else:
        return x

class Dist(Model):
    """
    Primarily, this class functions as a container for randomness required for distributions.
    """
    random_numbers = t.Size([])
    def __init__(self, size, **kwargs):
        super().__init__()
        self.z = RV((*self.random_numbers, *size))
        self.kwargs = kwargs

        # for easy access in future, and to ensure that they are recorded as modules,
        # record keyword arguments as fields
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, **kwargs):
        # combine kwargs provided at initialization and runtime
        ret = self.dist(
            self.z(), 
            **{k:unwrap(v) for k,v in self.kwargs.items()},
            **kwargs
        )
        self.prev = ret
        return ret

class Normal(Dist):
    dist = staticmethod(normal)
    
class LogNormal(Dist):
    dist = staticmethod(lognormal)

class Uniform(Dist):
    dist = staticmethod(uniform)

class Exponential(Dist):
    dist = staticmethod(exponential)

class Laplace(Dist):
    dist = staticmethod(laplace)

class Gumbel(Dist):
    dist = staticmethod(gumbel)

class Logistic(Dist):
    dist = staticmethod(gumbel)

class Delta(Dist):
    dist = staticmethod(delta)

class Pareto(Dist):
    dist = staticmethod(pareto)

class Gamma(Dist):
    random_numbers = t.Size([gamma_K+1])
    dist = staticmethod(gamma)

class LogGamma(Dist):
    random_numbers = t.Size([gamma_K+1])
    dist = staticmethod(loggamma)

class LogInvGamma(Dist):
    random_numbers = t.Size([gamma_K+1])
    dist = staticmethod(loginvgamma)

class InvGamma(Dist):
    random_numbers = t.Size([gamma_K+1])
    dist = staticmethod(invgamma)

class LogDirichlet(Dist):
    random_numbers = t.Size([gamma_K+1])
    dist = staticmethod(logdirichlet)

class Dirichlet(Dist):
    random_numbers = t.Size([gamma_K+1])
    dist = staticmethod(dirichlet)

class Beta(Dist):
    random_numbers = t.Size([2, gamma_K+1])
    dist = staticmethod(beta)
