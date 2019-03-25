import torch as t
import torch.nn as nn
import torch.nn.functional as F

#class VITensor(nn.Module):
class NormalVI(nn.Module):
    def __init__(self, rv):
        super().__init__()
        self.rv = rv
        self.mean = nn.Parameter(t.randn(rv.size))
        self.log_prec = nn.Parameter(8*t.ones(rv.size))
        
    def log_variance(self):
        return F.logsigmoid(-self.log_prec) # log(1 / (1+log_prec.exp()))

    def variance(self):
        return t.sigmoid(-self.log_prec)

    def std(self):
        return ( 0.5*self.log_variance()).exp()

    def inv_std(self):
        return (-0.5*self.log_variance()).exp()

    def rsample_kl(self):
        z = t.randn(self.rv.size)
        log_var = self.log_variance()

        scale = (0.5*log_var).exp()
        self.rv._value = self.mean + scale * z

        logp = self.rv.log_prob().sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return logq - logp

class VI(nn.Module):
    """
    Wrapper class that actually runs VI
    """
    def __init__(self, model, opt=t.optim.Adam, opt_kwargs={}):
        super().__init__()
        self.model = model
        assert self.model.is_conditioned

        self.vits = {}
        #### Look at all unconditioned random variables
        for k, v in self.model.named_uncond_rvs():
            vit = v.vi()
            self.vits[v] = vit
            setattr(self, "_"+k.replace(".", "_"), vit)

        self.opt = opt(self.parameters(), **opt_kwargs)

    def fit_one_step(self):
        self.zero_grad()
        elbo = self.elbo()
        (-elbo).backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for i in range(T):
            elbo = self.fit_one_step()
            if i % 1000 == 0:
                print(elbo.item())

        #### Detach rv._value from VI computation.
        for vit in self.vits.values():
            vit.rv._value.detach_()

    def elbo(self):
        total = 0.
        for vit in self.vits.values():
            total -= vit.rsample_kl()
        for v in self.model.cond_rvs():
            total += v.log_prob().sum()
        return total

    def __getitem__(self, key):
        return self.vits[key]
