#TODOs:
# mass matrix adaptation
# KE = p^T M^{-1} p
# \dot{x} = M^{-1} p
# sample p using covariance M
# M = E[g^2]

# effective-sample-size
# chat to Yul
# sample from the model
# define Chain class

# to think about
# convenient sampling from the generative model

import torch as t
import math
from torch.distributions import Normal
from timeit import default_timer as timer

t.set_default_tensor_type(t.FloatTensor)

class Trace:
    def __getitem__(self, name):
        return self.dict[name]

class SampleTrace(Trace):
    def __init__(self):
        self.dict = {}

    def sample(self, name, dist):
        assert name not in self.dict
        self.dict[name] = dist.sample()
        return None

    def obs(self, value, dist):
        pass


class LogProbTrace(Trace):
    def __init__(self, dict_):
        self.dict = dict_
        self.log_prob = 0.

    def sample(self, name, dist):
        self.log_prob += dist.log_prob(self.dict[name])
        return None

    def obs(self, value, dist):
        self.log_prob += dist.log_prob(value)
        return None


class JointDistribution():
    def sample(self):
        tr = SampleTrace()
        self.forward(tr)
        return tr.dict

    def log_prob(self, dict_):
        tr = LogProbTrace(dict_)
        self.forward(tr)
        return tr.log_prob

    def sample_log_prob(self):
        tr = SampleLogProbTrace()
        self.forward(tr)
        return tr.dict, tr.log_prob

class SimpleJD(JointDistribution):
    def forward(self, tr):
        tr.sample("x", Normal(0., 1.))
        tr.obs(4., Normal(tr["x"], 0.01))
        tr.sample("y", Normal(0., 1.))
        tr.obs(4., Normal(tr["y"], 1.))
        #Note: nothing is returned!




class HMC():
    def __init__(self, dist, rate, warmup, steps, trajectory_length):
        self.dist = dist
        self.rate = rate
        self.warmup = warmup
        self.steps = steps
        self.trajectory_length = trajectory_length


        
        self.x = dist.sample()
        self.results = dict_cpu_expand(steps, self.x)
        self.mass = dict_ones(self.x)
        self.mass_lambda = 0.1

    def step(self, adaptation=False):
        x_init = self.x

        # Proposal
        x_prop, accept_prob = LeapfrogIntegrator(self.dist, self.rate, self.trajectory_length, self.mass_lambda, self.mass, x_init).integrate(adaptation=adaptation)

        if t.rand(()).item() < accept_prob:
            self.x = x_prop

    def run(self):
        for _ in range(self.warmup):
            self.step(adaptation=True)

        for i in range(self.steps):
            self.step()
            dict_update(i, self.results, self.x)





class LeapfrogIntegrator():
    def __init__(self, dist, rate, trajectory_length, mass_lambda, mass, x_init):
        self.dist = dist
        self.rate = rate
        self.steps = int(trajectory_length // rate)
        self.mass_lambda = mass_lambda
        #Position
        self.x  = dict_clone_required_grad(x_init)
        #Momentum
        self.p  = dict_randn(x_init)
        #Squared gradient (for adaptation)
        self.g2 = dict_zeros(x_init)
        self.g = dict_zeros(x_init)

        #Convert position and momentum to lists for efficient iteration
        self.list_x = dict_flatten(self.x)
        self.list_p = dict_flatten(self.p)
        self.list_g2 = dict_flatten(self.g2)
        self.list_g = dict_flatten(self.g)
        self.list_mass = dict_flatten(mass)

        #Shape momentum initialization using mass matrix
        for i in range(len(self.list_p)):
            self.list_p[i].mul_(t.sqrt(self.list_mass[i]))

    def backward(self):
        # Zeros out past gradients
        for x in self.list_x:
            if x.grad is not None:
                x.grad.fill_(0.)

        # Computes new gradients
        lp = self.dist.log_prob(self.x)
        lp.backward()

        # Record moments of the gradient
        for i in range(len(self.list_x)):
            grad = self.list_x[i].grad
            self.list_g2[i].addcmul_(1/self.steps, grad, grad)
            self.list_g[i].add_(1/self.steps, grad)

        # Return log-probability (which is occasionally useful)
        return lp

    def position_step(self):
        for i in range(len(self.list_x)):
            #self.list_x[i].data.add_(self.rate, self.list_p[i])
            self.list_x[i].data.addcdiv_(self.rate, self.list_p[i], self.list_mass[i])

    def momentum_step(self):
        self.backward()
        for i in range(len(self.list_x)):
            self.list_p[i].add_(self.rate, self.list_x[i].grad)

    def momentum_half_step(self):
        lp = self.backward()
        for i in range(len(self.list_x)):
            self.list_p[i].add_(self.rate/2, self.list_x[i].grad)
        return lp

    def integrate(self, adaptation=False):
        # Compute initial log-probability (including initial half momemtum step)
        lp_p = 0.
        for i in range(len(self.list_p)):
            lp_p += -t.sum(self.list_p[i]**2/self.list_mass[i])/2
        lp_x = self.momentum_half_step()
        lp_init = lp_x + lp_p

        # Integration
        for _ in range(self.steps-1):
            self.position_step()
            self.momentum_step()
        self.position_step()

        # Compute final log-probability (including final half momemtum step)
        lp_x = self.momentum_half_step()
        lp_p = 0.
        for i in range(len(self.list_p)):
            lp_p += -t.sum(self.list_p[i]**2/self.list_mass[i])/2
        lp_prop = lp_x + lp_p

        acceptance_prob = (lp_prop - lp_init).exp()

        # adaptation
        if adaptation:
            for i in range(len(self.list_mass)):
                self.list_mass[i].mul_(1-self.mass_lambda).add_(self.mass_lambda, self.list_g2[i] - self.list_g[i]**2)

        return self.x, acceptance_prob




#### Utility functions:
def validate_dicts(*ds):
    for d1 in ds:
        #all the inputs are dicts
        assert isinstance(d1, dict)

        for d2 in ds:
            if d1 is not d2:
                for key in d1:
                    #all keys in d1 exist in d2
                    assert key in d2
                    assert type(d1[key]) == type(d2[key])

def dict_recurse_one(fn, d):
    output_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            output_dict[k] = dict_recurse_one(fn, v)
        else:
            output_dict[k] = fn(v)
    return output_dict

def dict_inplace_recurse(fn, *ds):
    validate_dicts(*ds)
    output_dict = {}
    for key in ds[0]:
        vs = [d[key] for d in ds]
        if isinstance(vs[0], dict):
            dict_recurse(fn, *vs)
        else:
            fn(*vs)

def dict_sum(fn, d):
    total = 0.
    for k, v in d.items():
        if isinstance(v, dict):
            total += dict_sum(fn, v)
        else:
            total += fn(v)
    return total

#### Recursions

def randn(x):
    return t.randn(x.shape) 
def dict_randn(d):
    return dict_recurse_one(randn, d)

def zeros(x):
    return t.zeros(x.shape)
def dict_zeros(d):
    return dict_recurse_one(zeros, d)

def ones(x):
    return t.ones(x.shape)
def dict_ones(d):
    return dict_recurse_one(ones, d)

def clone(x):
    return x.clone()
def dict_clone(d):
    return dict_recurse_one(clone, d)

def clone_required_grad(x):
    return x.detach().clone().requires_grad_()
def dict_clone_required_grad(d): 
    return dict_recurse_one(clone_required_grad, d)

def grad_(x):
    return x.grad
def dict_grad(d):
    return dict_recurse_one(grad_, d)

def dict_cpu_expand(N, d):
    fn = lambda x: t.zeros((N, *x.shape), device="cpu")
    return dict_recurse_one(fn, d)

#### Flatten

#def dict_iter(d):
#    for k, v in d.items():
#        if isinstance(v, dict):
#            for sub_v in dict_iter(v):
#                yield(sub_v)
#        else:
#            yield(v)
#
#def dicts_iter(d1, d2):
#    for k, v in d.items():
#        if isinstance(v, dict):
#            for sub_v in dict_iter(v):
#                yield(sub_v)
#        else:
#            yield(v)

def dict_flatten(d):
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result += dict_flatten(d)
        else:
            result.append(v)
    return result

def add_(mult, xs, ys):
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        x.data.add_(mult, y)
           
#### In-place ops

def dict_update(i, xs, ys):
    def update(x, y):
        x[i, ...] = y
    return dict_inplace_recurse(update, xs, ys)


jd = SimpleJD()
tr = jd.sample()
lp = jd.log_prob(tr)

start = timer()
hmc = HMC(jd, 1E-2, 100, 100, 2*math.pi)
hmc.run()
end = timer()
print(end-start)
