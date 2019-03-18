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
    return loc + scale * z

def lognormal(z, mu, sigma):
    return (mu + sigma * z).exp()

def uniform(z, a, b):
    return a + (b-a)*cdf(z)

def exponential(z, rate):
    return -log_ccdf(z)/rate

def laplace(z, loc, scale):
    sl = t.where(
        z < 0,
           math.log(2) + log_cdf(z),
        - (math.log(2) + log_ccdf(z))
    )
    return loc + scale*sl

def gumbel(z, mu, beta):
    return mu - beta*t.log(-log_cdf(z))

def logistic(z, loc, scale):
    return loc + scale * (log_cdf(z) - log_ccdf(z))

def trans(z, dist):
    """
    Allows us to use:
    Cauchy
    Half Cauchy
    Half Normal
    """
    return dist.icdf(cdf(z))

zs = t.arange(-2, 3.)
assert t.allclose(normal(zs, 1., 2.),    trans(zs, Normal(1., 2.)), rtol=1E-4, atol=1E-4)
assert t.allclose(lognormal(zs, 1., 2.), trans(zs, t.distributions.LogNormal(1., 2.)), rtol=1E-4, atol=1E-4)
assert t.allclose(uniform(zs, 1., 3.),   trans(zs, t.distributions.Uniform(1., 3.)), rtol=1E-4, atol=1E-4)
assert t.allclose(exponential(zs, 2.),   trans(zs, t.distributions.Exponential(2.)), rtol=1E-4, atol=1E-4)
assert t.allclose(laplace(zs, 1., 3.),   trans(zs, t.distributions.Laplace(1., 3.)), rtol=1E-4, atol=1E-4)
assert t.allclose(gumbel(zs, 1., 3.),    trans(zs, t.distributions.Gumbel(1., 3.)), rtol=1E-4, atol=1E-4)
assert t.allclose(logistic(t.randn(10**6), 0., 1.).var(),  t.Tensor([math.pi**2/3]), rtol=1E-3, atol=1E-3)

from bb import RV
class Normal(RV):
    def forward(self, loc, scale):
        return normal(self.x, loc, scale)

class LogNormal(RV):
    def forward(self, mu, sigma):
        return lognormal(self.x, mu, sigma)

class Uniform(RV):
    def forward(self, a, b):
        return uniform(self.x, a, b)

class Exponential(RV):
    def forward(self, rate):
        exponential(self.x, rate)

class Laplace(RV):
    def forward(self, loc, scale):
        laplace(self.x, loc, scale)

class Gumbel(RV):
    def forward(self, mu, beta):
        gumbel(self.x, mu, beta)

class Logistic(RV):
    def forward(self, loc, scale):
        logistic(self.x, loc, scale)
