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
        self.x = t.randn(self.size)

    def __call__(self):
        return self.x

    #### VI

    def vi_init(self):
        self.vi_mean = nn.Parameter(t.randn(self.size))
        self.vi_log_prec = nn.Parameter(t.Tensor(self.size).fill_(8.))

    def vi_log_variance(self):
        return F.logsigmoid(-self.vi_log_prec) # log(1 / (1+log_prec.exp()))

    def vi_variance(self):
        return t.sigmoid(-self.vi_log_prec)

    def vi_std(self):
        return ( 0.5*self.vi_log_variance()).exp()

    def vi_inv_std(self):
        return (-0.5*self.vi_log_variance()).exp()

    def vi_rsample_kl(self):
        z = t.randn(self.size)
        log_var = self.vi_log_variance()

        scale = (0.5*log_var).exp()
        self.x = self.vi_mean + scale * z

        logp = - 0.5 * (self.x**2).sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return logq - logp

    #### HMC

    def hmc_init(self, chain_length):
        self.hmc_inv_mass = self.vi_variance()
        self.hmc_sqrt_mass = self.vi_inv_std()

        #state of Markov chain 
        self.hmc_x_chain = self.vi_mean.detach().clone()

        #state of leapfrog integrator
        self.x = self.hmc_x_chain.clone().requires_grad_()
        self.hmc_p = t.zeros(self.size)

        self.hmc_samples = t.zeros(t.Size([chain_length]) + self.size, device="cpu")

        assert not self.hmc_x_chain.requires_grad
        assert     self.x.requires_grad
        assert not self.hmc_p.requires_grad

    def hmc_momentum_step(self, rate):
        self.hmc_p.add_(rate, self.x.grad)
        self.hmc_p.add_(-rate, self.x.data)

    def hmc_position_step(self, rate):
        self.x.data.addcmul_(rate, self.hmc_inv_mass, self.hmc_p)

    def hmc_zero_grad(self):
        if self.x.grad is not None:
            self.x.grad.fill_(0.)

    def hmc_log_prior_xp(self):
        lp_x = -0.5*(self.x**2).sum()
        lp_p = -0.5*(self.hmc_inv_mass*self.hmc_p**2).sum()
        return lp_x + lp_p

    def hmc_record_sample(self, i):
        self.hmc_samples[i,...] = self.hmc_x_chain

    def hmc_accept(self):
        self.hmc_x_chain.fill_(self.x)

    def hmc_refresh_momentum(self):
        self.hmc_p.normal_(0., 1.)
        self.hmc_p.mul_(self.hmc_sqrt_mass)


class Model(nn.Module):
    """
    Overload: 
    __init__ 
    __call__
    """

    def randn(self):
        for v in self._modules.values():
            v.randn()

    def forward(self, obs):
        """
        Typically, redefine models to provide an output.
        The outer Model should provide a log-probability when called.
        Here is a fallback to allow
        """
        return self.likelihood(*args, **kwargs).log_prob(obs).sum()

    def rvs(self):
        return (mod for mod in self.modules() if isinstance(mod, RV))

    ##### VI

    #def vi_init(self):
    #    for v in self._modules.values():
    #        v.vi_init()

    #def vi_rsample_kl(self):
    #    total = 0.
    #    for v in self._modules.values():
    #        total += v.vi_rsample_kl()
    #    return total

    ##### HMC
    #def hmc_init(self, chain_length):
    #    for v in self._modules.values():
    #        v.hmc_init(chain_length)

    #def hmc_accept(self):
    #    for v in self._modules.values():
    #        v.hmc_accept()

    #def hmc_record_sample(self, i):
    #    for v in self._modules.values():
    #        v.hmc_record_sample(i)

    #def hmc_zero_grad(self):
    #    for v in self._modules.values():
    #        v.hmc_zero_grad()

    #def hmc_position_step(self, rate):
    #    for v in self._modules.values():
    #        v.hmc_position_step(rate)

    #def hmc_refresh_momentum(self):
    #    for v in self._modules.values():
    #        v.hmc_refresh_momentum()

    #def hmc_momentum_step(self, rate):
    #    for v in self._modules.values():
    #        v.hmc_momentum_step(rate)

    #def hmc_log_prior_xp(self):
    #    total = 0.
    #    for v in self._modules.values():
    #        total += v.hmc_log_prior_xp()
    #    return total



class VI():
    """
    Wrapper class that actually runs VI
    """
    def __init__(self, model, opt=t.optim.Adam, opt_kwargs={}):
        super().__init__()
        self.model = model
        for rv in self.model.rvs():
            rv.vi_init()
        self.opt = opt(self.model.parameters(), **opt_kwargs)

    def fit_one_step(self):
        self.model.zero_grad()
        kl = self.rsample_kl()
        elbo = self.model() - kl
        loss = -elbo
        loss.backward()
        self.opt.step()
        return elbo

    def fit(self, T=1000):
        for _ in range(T):
            elbo = self.fit_one_step()

    def rsample_kl(self):
        total = 0.
        for rv in self.model.rvs():
            total += rv.vi_rsample_kl()
        return total

class HMC():
    def __init__(self, model, chain_length, warmup=0, rate=1E-2, trajectory_length=1.):
        self.model = model
        model.hmc_init(chain_length)

        self.chain_length = chain_length
        self.warmup = warmup
        self.rate = rate
        self.steps = int(trajectory_length // rate)

    def position_step(self, rate):
        self.model.hmc_position_step(rate)

    def momentum_step(self, rate):
        self.model.hmc_zero_grad()
        lp = self.model()
        lp.backward()
        self.model.hmc_momentum_step(rate)
        return lp


    def step(self, i=None):
        self.model.hmc_refresh_momentum()

        lp_prior_xp = self.model.hmc_log_prior_xp()
        lp_like     = self.momentum_step(0.5*self.rate)
        lp_init     = lp_prior_xp + lp_like

        # Integration
        for _ in range(self.steps-1):
            self.position_step(self.rate)
            self.momentum_step(self.rate)
        self.position_step(self.rate)

        lp_like     = self.momentum_step(0.5*self.rate)
        lp_prior_xp = self.model.hmc_log_prior_xp()
        lp_prop     = lp_prior_xp + lp_like

        acceptance_prob = (lp_prop - lp_init).exp()

        #Acceptance
        if t.rand(()) < acceptance_prob:
            self.model.hmc_accept()

        if i is not None:
            self.model.hmc_record_sample(i)

    def run(self):
        for _ in range(self.warmup):
            self.step()

        for i in range(self.chain_length):
            self.step(i)

