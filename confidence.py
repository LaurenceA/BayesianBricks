#BetaBernoulli
#BetaBinomial
#DirichletCategorical
#DirichletMultinomial
#
#GammaNormal
import math
import torch as t
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

def confidence(zs, thresholds, beta):
    """
    take a Logistic distribution, centered around z, with an inverse-width, beta
    """
    assert 1 == zs.size(-1)
    diff = beta*(thresholds - zs)

    u = diff[..., 1:]
    l = diff[..., :-1]

    lp =  F.logsigmoid(u) + F.logsigmoid(l) - l + t.log1p(-t.exp(l-u))
    lp = t.cat([
        F.logsigmoid(diff[..., 0:1]), 
        lp,
        F.logsigmoid(-diff[..., -1:])
    ], dim=-1)

    return OneHotCategorical(logits=lp)

    #size = diff[..., 0:1].size()
    #upper_diff = t.cat([diff, math.inf*t.ones(size)], dim=-1)
    #lower_diff = t.cat([-math.inf*t.ones(size), diff], dim=-1)
    #val = t.sigmoid(upper_diff) - t.sigmoid(lower_diff)



ret = confidence(1.3*t.ones(1,1), t.arange(-3, 4.), 40.)
