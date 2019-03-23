import torch as t
import torch.nn as nn
import torch.nn.functional as F

from rv import RV


class VITensor(nn.Module):
    def __init__(self, rv):
        super().__init__()
        self.rv = rv
        self.mean = nn.Parameter(t.randn(rv.size))
        self.log_prec = nn.Parameter(t.Tensor(rv.size).fill_(8.))
        
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

        logp = - 0.5 * (self.rv._value**2).sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return logq - logp

class VI(nn.Module):
    """
    Wrapper class that actually runs VI
    """
    def __init__(self, model, opt=t.optim.Adam, opt_kwargs={}):
        super().__init__()
        self.model = model
        self.vits = []

        for k, v in self.model.named_modules():
            if isinstance(v, RV):
                vit = VITensor(v)
                self.vits.append(vit)
                self.add_module("_"+k.replace(".", "_"), vit)

        self.opt = opt(self.parameters(), **opt_kwargs)

    def fit_one_step(self):
        self.zero_grad()
        kl = self.rsample_kl()
        elbo = self.model() - kl
        loss = -elbo
        loss.backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for i in range(T):
            elbo = self.fit_one_step()
            if i % 1000 == 0:
                print(elbo.item())

    def rsample_kl(self):
        total = 0.
        for vit in self.vits:
            total += vit.rsample_kl()
        return total
