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
        self.x = t.randn(size)

    def randn(self):
        self.x = randn(self.size)

    def __call__(self):
        return self.value

    #### VI

    def vi_init(self):
        self.mean = nn.Parameter(t.randn(self.size))
        self.log_prec = nn.Parameter(t.Tensor(self.size).fill_(8.))

    def vi_log_variance(self):
        return F.logsigmoid(-self.log_prec) # log(1 / (1+log_prec.exp()))

    def vi_variance(self):
        return t.sigmoid(-self.log_prec)

    def vi_std(self):
        return ( 0.5*self.vi_log_variance()).exp()

    def vi_inv_std(self):
        return (-0.5*self.vi_log_variance()).exp()

    def vi_rsample_kl(self):
        z = t.randn(self.size)
        log_var = self.vi_log_variance()

        scale = (0.5*log_var).exp()
        self.x = self.mean + scale * z

        logp = - 0.5 * (self.x**2).sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return logq - logp

    #### HMC

    def init_HMC(self, chain_length):
        self.inv_mass = vit.variance()
        self.sqrt_mass = vit.inv_std()

        #state of Markov chain 
        self.x_mcmc = t.zeros(size)

        #state of leapfrog integrator
        self.x = self.x_mcmc.clone().requires_grad_()
        self.p = t.zeros(vit.size)

        assert not self.x_mcmc.requires_grad
        assert     self.x.requires_grad
        assert not self.p.requires_grad


class RVs(nn.Module):
    """
    Overload: 
    __init__ 
    __call__
    """

    def init_vi(self):
        self.mean = nn.Parameter(t.randn(self.size))
        self.log_prec = nn.Parameter(t.Tensor(self.size).fill_(8.))

    def randn(self):
        for v in self._modules.values():
            v.randn()
                
    #### VI

    def vi_init(self):
        for v in self._modules.values():
            v.vi_init()

    def vi_rsample_kl(self):
        total = 0.
        for v in self._modules.values():
            total += v.vi_rsample_kl()
        return total

class Normal(RV):
    def __call__(self, loc, scale):
        return loc + scale*self.x

class Model(RVs):
    def __init__(self):
        super().__init__()
        self.a = Normal(())
        self.b = Normal(())

    def __call__(self):
        a = self.a(0., 1.)
        b = self.b(0., 1.)
        mean = t.stack([a, b])
        scale = t.Tensor([1., 0.01])
        return t.distributions.Normal(mean, scale).log_prob(t.zeros(2)).sum()


class VI():
    """
    Wrapper class that actually runs VI
    """
    def __init__(self, model, opt=t.optim.Adam, opt_kwargs={}):
        super().__init__()
        model.vi_init()
        self.model = model
        self.opt = opt(self.model.parameters(), **opt_kwargs)

    def fit_one_step(self):
        self.model.zero_grad()
        kl = self.model.vi_rsample_kl()
        elbo = self.model() - kl
        loss = -elbo
        loss.backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for _ in range(T):
            elbo = self.fit_one_step()



m = Model()
m()

m.vi_init()
m.vi_rsample_kl()

vi = VI(m)
vi.fit(3*10**4)


