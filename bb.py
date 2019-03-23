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
        self._value = t.randn(size)

    def randn(self):
        self._value = t.randn(self.size)

    def __call__(self):
        return self._value

    def log_prior(self):
        return -0.5*(self._value**2).sum()

    #### VI


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
        self._value = self.vi_mean + scale * z

        logp = - 0.5 * (self._value**2).sum()
        logq = - 0.5 * (z**2 + log_var).sum()
        return logq - logp



    #### HMC

    def hmc_momentum_step(self, rate):
        self.hmc_p.add_(rate, self._value.grad)
        self.hmc_p.add_(-rate, self._value.data)

    def hmc_position_step(self, rate):
        self._value.data.addcmul_(rate, self.hmc_inv_mass, self.hmc_p)

    def hmc_zero_grad(self):
        if self._value.grad is not None:
            self._value.grad.fill_(0.)

    def hmc_log_prior_xp(self):
        lp_x = -0.5*(self._value**2).sum()
        lp_p = -0.5*(self.hmc_inv_mass*self.hmc_p**2).sum()
        return lp_x + lp_p

    def hmc_accept(self):
        self.hmc_x_chain.copy_(self._value)

    def hmc_step_initialize(self):
        self.hmc_p.normal_(0., 1.)
        self.hmc_p.mul_(self.hmc_sqrt_mass)
        self._value.data.copy_(self.hmc_x_chain)


class Model(nn.Module):
    """
    Overload: 
    __init__ 
    __call__
    """
    def rvs(self):
        return (mod for mod in self.modules() if isinstance(mod, RV))

    def models(self):
        return (mod for mod in self.modules() if isinstance(mod, Model))

    def refresh(self):
        for rv in self.rvs():
            rv.randn()

    def dump(self):
        result = {}
        for k, v in self.named_modules():
            if hasattr(v, "_value"):
                result[k] = v._value
            else:
                result[k] = None
        return result

def sum_all_except(x, ind_dims):
    for dim in ind_dims:
        assert 1 < x.size(dim)

    sum_dims = set(range(len(x.size()))) - set(ind_dims)
    return x.sum(dim=tuple(sum_dims), keepdim=True)

class MCMC():
    """
    Assumes that the ratio of proposals forward and back is equal,
    which is true for symmetric proposals, and HMC.

    Critical technical detail:
    the ind_dims, and log_like must be compatible with all log_priors.
    """
    def __init__(self, rvs, ind_dims=()):
        self.rvs = list(rvs)
        for rv in self.rvs:
            assert isinstance(rv, RV)
        self.ind_dims = ind_dims

        self.init_proposal()

    def init_proposal(self):
        pass

    def proposal(self):
        raise NotImplementedError()

    def log_prior(self):
        total = 0.
        for rv in self.rvs:
            total += sum_all_except(rv.log_prior(), self.ind_dims)
        return total


    def step(self, model):
        xs_prev = [rv._value.clone() for rv in self.rvs]
        lp_prev = model() + self.log_prior()

        self.proposal()

        xs_next = [rv._value.clone() for rv in self.rvs]
        lp_next = model() + self.log_prior()
        
        lp_diff = sum_all_except(lp_next - lp_prev, self.ind_dims)
        accept_prob = lp_diff.exp()
        accept_cond = t.rand(accept_prob.size()) < accept_prob

        for i in range(len(self.rvs)):
            new_value = t.where(
                accept_cond,
                xs_next[i],
                xs_prev[i]
            )
            assert new_value.size() == self.rvs[i].size
            self.rvs[i]._value = new_value

class MHMC(MCMC):
    def proposal(self):
        for rv in self.rvs:
            rv._value.add_(0.3*rv.vi_std()*t.randn(rv.size))
            #rv._value.add_(0.03*t.randn(rv.size))


class Chain():
    """
    Manages the recording of MCMC samples
    """
    def __init__(self, model, kernels, chain_length, warmup=0):
        self.model = model
        self.kernels = kernels
        self.chain_length = chain_length
        self.warmup = warmup

        #Initialize recording buffers
        model()
        for m in self.model.modules():
            if hasattr(m, "_value"):
                m._mcmc_samples = t.zeros(t.Size([chain_length]) + m._value.size(), device="cpu")

    def run(self):
        for i in range(self.warmup):
            for kernel in self.kernels:
                kernel.step(self.model)

        for i in range(self.chain_length):
            for kernel in self.kernels:
                kernel.step(self.model)

            #Record current sample
            for m in self.model.modules():
                if hasattr(m, "_value"):
                    m._mcmc_samples[i,...] = m._value


class VI():
    """
    Wrapper class that actually runs VI
    """
    def __init__(self, model, opt=t.optim.Adam, opt_kwargs={}):
        super().__init__()
        self.model = model
        self.vi_init()
        self.opt = opt(self.model.parameters(), **opt_kwargs)

    def vi_init(self):
        for rv in self.model.rvs():
            rv.vi_mean = nn.Parameter(t.randn(rv.size))
            rv.vi_log_prec = nn.Parameter(t.Tensor(rv.size).fill_(8.))

    def fit_one_step(self):
        self.model.zero_grad()
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
        for rv in self.model.rvs():
            total += rv.vi_rsample_kl()
        return total

class HMC():
    def __init__(self, model, chain_length, warmup=0, rate=1E-2, trajectory_length=1.):
        self.model = model
        self.hmc_init(chain_length)

        self.chain_length = chain_length
        self.warmup = warmup
        self.rate = rate
        self.steps = int(trajectory_length // rate)

    def hmc_init(self, chain_length):
        for m in self.model.modules():
            if hasattr(m, "_value"):
                m.hmc_samples = t.zeros(t.Size([chain_length]) + m._value.size(), device="cpu")

            if isinstance(m, RV):
                m.hmc_inv_mass = m.vi_variance()
                m.hmc_sqrt_mass = m.vi_inv_std()

                #state of Markov chain 
                m.hmc_x_chain = m.vi_mean.detach().clone()

                #state of leapfrog integrator
                m._value = m.hmc_x_chain.clone().requires_grad_()
                m.hmc_p = t.zeros(m.size)

                assert not m.hmc_x_chain.requires_grad
                assert     m._value.requires_grad
                assert not m.hmc_p.requires_grad

    def position_step(self, rate):
        for rv in self.model.rvs():
            rv.hmc_position_step(rate)

    def momentum_step(self, rate):
        self.hmc_zero_grad()
        lp = self.model()
        lp.backward()
        for rv in self.model.rvs():
            rv.hmc_momentum_step(rate)
        return lp

    def accept(self):
        for rv in self.model.rvs():
            rv.hmc_accept()

    def record_sample(self, i):
        for m in self.model.modules():
            if hasattr(m, "_value"):
                m.hmc_samples[i,...] = m._value

    def hmc_zero_grad(self):
        for rv in self.model.rvs():
            rv.hmc_zero_grad()

    def step_initialize(self):
        for rv in self.model.rvs():
            rv.hmc_step_initialize()

    def log_prior_xp(self):
        total = 0.
        for rv in self.model.rvs():
            total += rv.hmc_log_prior_xp()
        return total

    def step(self, i=None):
        self.step_initialize()

        lp_prior_xp = self.log_prior_xp()
        lp_like     = self.momentum_step(0.5*self.rate)
        lp_init     = lp_prior_xp + lp_like
        
        #Record sample here, because 
        #  _value is set to last sample in the MCMC chain, and is not updated by momentum_step
        #  model has just been run inside momentum_step (so all intermediate _value) are correct
        if i is not None:
            self.record_sample(i)

        # Integration
        for _ in range(self.steps-1):
            self.position_step(self.rate)
            self.momentum_step(self.rate)
        self.position_step(self.rate)

        lp_like     = self.momentum_step(0.5*self.rate)
        lp_prior_xp = self.log_prior_xp()
        lp_prop     = lp_prior_xp + lp_like

        acceptance_prob = (lp_prop - lp_init).exp()

        #Acceptance

        accept = t.rand(()) < acceptance_prob
        if accept:
            self.accept()
        return accept

    def run(self):
        accepts = 0
        iters = 0
        for _ in range(self.warmup):
            self.step()

        for i in range(self.chain_length):
            self.step(i)
            accepts += self.step()
            iters += 1
        return accepts / iters

