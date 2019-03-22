from unittest import TestCase

import torch as t
from bb import RV, Model, VI, HMC
from distributions import Normal

def between(x, a, b):
    return (a < x) and (x < b)

class DifferentScales(Model):
    def __init__(self):
        super().__init__()
        self.a = Normal((), loc=t.zeros(()), scale=t.ones(()))
        self.b = Normal((), loc=t.zeros(()), scale=t.ones(()))

    def __call__(self):
        a = self.a()
        b = self.b()
        mean = t.stack([a, b])
        scale = t.Tensor([1., 0.01])
        return t.distributions.Normal(mean, scale).log_prob(t.zeros(2)).sum()

class VI_Integration(TestCase):
    def runTest(self):
        m = DifferentScales()
        vi = VI(m)
        vi.fit(3*10**4)

        var_a = m.a.z.vi_variance().item()
        self.assertTrue(between(var_a, 0.4, 0.6), "VI variance of a was: " + str(var_a) + " should have been: " + str(0.5))

        var_b = m.b.z.vi_variance().item()
        self.assertTrue(between(var_b, 0.75*1E-4, 1.25E-4), "VI variance of a was: " + str(var_b) + " should have been: " + str(0.0001))

        hmc = HMC(m, 500)
        hmc.run()

        var_a = m.a.z.hmc_samples.var().item()
        self.assertTrue(between(var_a, 0.4, 0.6), "HMC variance of a was: " + str(var_a) + " should have been: " + str(0.5))

        var_b = m.b.z.hmc_samples.var().item()
        self.assertTrue(between(var_b, 0.75*1E-4, 1.25E-4), "HMC variance of a was: " + str(var_b) + " should have been: " + str(0.0001))
