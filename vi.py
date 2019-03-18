import torch as t
import torch.nn as nn
import torch.nn.functional as F

class VITensor(nn.Module):
    def __init__(self, mtensor, batch_size=t.Size([])):
        super().__init__()
        self.mtensor=mtensor
        self.size = mtensor.size
        self.mean = nn.Parameter(t.randn(size))
        self.log_prec = nn.Parameter(t.Tensor(size).fill_(8.))

    def log_variance(self):
        return F.logsigmoid(-self.log_prec) # log(1 / (1+log_prec.exp()))

    def variance(self):
        return t.sigmoid(-self.log_prec)

    def std(self):
        return ( 0.5*self.log_variance()).exp()

    def inv_std(self):
        return (-0.5*self.log_variance()).exp()

    def rsample_kl(self, batch_size=t.Size([])):
        z = t.randn(batch_size + self.size)
        log_var = self.log_variance()

        scale = (0.5*log_var).exp()
        x = self.mean + scale * z

        logp = - 0.5 * (x**2).sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return x, logq - logp

class VIDict(nn.Module):
    def __init__(self, mdict):
        assert isinstance(sizes, dict)

        super().__init__()
        for key, val in mdict.tensors.items():
            if isinstance(val, dict):
                setattr(self, key, VIDict(val))
            else:
                setattr(self, key, VITensor(val))

    def rsample(self):
                
    def rsample_kl(self):
        result_dict = {}
        total_kl = 0.
        for k, v in self._modules.items():
            result_dict[k], kl = v.rsample_kl()
            total_kl += kl
        return result_dict, total_kl
        

class VI():
    def __init__(self, model, batch_size=t.Size([]), opt=t.optim.Adam, opt_kwargs={}):
        super().__init__()
        self.tensors = VIDict(model)
        self.opt = opt(self.tensors.parameters(), **opt_kwargs)

    def fit_one_step(self):
        self.tensors.zero_grad()
        rsample, kl = self.tensors.rsample_kl()
        elbo = self.fn(rsample) - kl
        loss = -elbo
        loss.backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for _ in range(T):
            elbo = self.fit_one_step()
