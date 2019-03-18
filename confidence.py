#BetaBernoulli
#BetaBinomial
#DirichletCategorical
#DirichletMultinomial
#
#GammaNormal
import torch as t
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

def confidence(zs, thresholds, beta):
    """
    take a Logistic distribution, centered around z, with an inverse-width, beta
    """
    assert 1 == zs.size(-1)
    diff = thresholds - zs
    cdfs = t.sigmoid(beta*diff)

    size = cdfs[..., 0:1].size()

    lower_cdfs = t.cat([t.zeros(size), cdfs], dim=-1)
    upper_cdfs = t.cat([cdfs, t.ones(size)], dim=-1)

    return upper_cdfs - lower_cdfs

ret = confidence(1.3*t.ones(1,1), t.arange(-3, 4.), 5.)
