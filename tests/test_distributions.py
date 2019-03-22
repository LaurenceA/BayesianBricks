from unittest import TestCase
import math
import torch as t
import distributions as d

zs = t.arange(-2, 3.)

class Normal(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.normal(zs, 1., 2.), 
                d.trans(zs, t.distributions.Normal(1., 2.)), 
                rtol=1E-4, atol=1E-4
            ),
            "Normal"
        )

class LogNormal(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.lognormal(zs, 1., 2.), 
                d.trans(zs, t.distributions.LogNormal(1., 2.)), 
                rtol=1E-4, atol=1E-4
            ),
            "LogNormal"
        )

class Uniform(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.uniform(zs, 1., 3.), 
                d.trans(zs, t.distributions.Uniform(1., 3.)), 
                rtol=1E-4, atol=1E-4
            ),
            "Uniform"
        )

class Exponential(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.exponential(zs, 2.), 
                d.trans(zs, t.distributions.Exponential(2.)), 
                rtol=1E-4, atol=1E-4
            ),
            "Exponential"
        )

class Laplace(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.laplace(zs, 1., 3.), 
                d.trans(zs, t.distributions.Laplace(1., 3.)), 
                rtol=1E-4, atol=1E-4
            ),
            "Laplace"
        )

class Gumbel(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.gumbel(zs, 1., 3.), 
                d.trans(zs, t.distributions.Gumbel(1., 3.)), 
                rtol=1E-4, atol=1E-4
            ),
            "Gumbel"
        )

class Gumbel(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.gumbel(zs, 1., 3.), 
                d.trans(zs, t.distributions.Gumbel(1., 3.)), 
                rtol=1E-4, atol=1E-4
            ),
            "Gumbel"
        )

class Logistic(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.logistic(t.randn(10**6), 0., 1.).var(),
                t.Tensor([math.pi**2/3]),
                rtol=1E-2
            ),
            "Logistic"
        )

class Pareto(TestCase):
    def runTest(self):
        self.assertTrue(
            t.allclose(
                d.pareto(t.randn(10**6), 3., 2.).mean(),
                t.Tensor([3.]),
                rtol=1E-2
            ),
            "Pareto"
        )
