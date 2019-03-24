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

