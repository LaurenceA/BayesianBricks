#TODOs:
# separate mcmc_steps and integration_steps
# mass matrix (flexible adaptation based on squared-gradients)
# effective-sample-size
# chat to Yul
# sample from the model
# optimize dictionary passes
# define Chain class

# to think about
# convenient sampling from the generative model

import torch as t
from torch.distributions import Normal

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
        tr.obs(4., Normal(tr["x"], 0.1))
        #Note: nothing is returned!




class HMC():
    def __init__(self, dist, rate, warmup, steps):
        self.dist = dist
        self.rate = rate
        self.warmup = warmup
        self.steps = steps

        self.x = dist.sample()
        self.results = dict_cpu_expand(steps, self.x)

    def log_prob(self, xs, ps):
        lp_x = self.dist.log_prob(xs)
        lp_p = dict_sum(lambda p: t.sum(-p*p/2), ps)
        return lp_x + lp_p

    def step(self):
        x_init = self.x
        p_init = dict_randn(x_init)

        # Proposal
        x_prop, p_prop = LeapfrogIntegrator(self.dist, self.rate, self.steps, x_init, p_init).integrate()

        # Accept/reject
        lp_init = self.log_prob(x_init, p_init)
        lp_prop = self.log_prob(x_prop, p_prop)

        accept_prob = t.exp(lp_prop-lp_init).item()
        if t.rand(()).item() < accept_prob:
            self.x = x_prop

    def run(self):
        for _ in range(self.warmup):
            self.step()

        for i in range(self.steps):
            self.step()
            dict_update(i, self.results, self.x)





class LeapfrogIntegrator():
    def __init__(self, dist, rate, steps, x_init, p_init):
        self.dist = dist
        self.rate = rate
        self.steps = steps
        #Position
        self.x = dict_clone_required_grad(x_init)
        #Momentum
        self.p = dict_clone(p_init)

    def grad(self):
        dict_zero_grad(self.x)
        lp = self.dist.log_prob(self.x)
        lp.backward()
        return dict_grad(self.x)

    def position_step(self):
        dict_add_(self.rate, self.x, self.p)

    def momentum_step(self):
        dict_add_(self.rate, self.p, self.grad())

    def momentum_half_step(self):
        dict_add_(self.rate/2, self.p, self.grad())

    def integrate(self):
        self.momentum_half_step()
        for _ in range(self.steps-1):
            self.position_step()
            self.momentum_step()
        self.position_step()
        self.momentum_half_step()
        return self.x, self.p




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

def clone(x):
    return x.clone()
def dict_clone(d):
    return dict_recurse_one(clone, d)

def clone_required_grad(x):
    return x.detach().clone().requires_grad_()
def dict_clone_required_grad(d): 
    return dict_recurse_one(clone_required_grad, d)

def grad(x):
    return x.grad
def dict_grad(d):
    return dict_recurse_one(grad, d)

def dict_cpu_expand(N, d):
    fn = lambda x: t.zeros((N, *x.shape), device="cpu")
    return dict_recurse_one(fn, d)

#### Flatten

def dict_flatten(d):
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result += dict_flatten(d)
        else
            result.append(v)
    return result

def zero_grad(xs):
    for x in xs:
        if x.grad is not None:
            x.grad.fill_(0.)

def add_(mult, xs, ys):
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        x.data.add_(mult, y)
           
#### In-place ops

#def zero_grad(x):
#    if x.grad:
#        x.grad.fill_(0.)
#def dict_zero_grad(input_dict):
#    dict_inplace_recurse(zero_grad, input_dict)
#
#def dict_add_(mult, xs, ys):
#    fn = lambda x, y: x.data.add_(mult, y)
#    return dict_inplace_recurse(fn, xs, ys)

def dict_update(i, xs, ys):
    def update(x, y):
        x[i, ...] = y
    return dict_inplace_recurse(update, xs, ys)


jd = SimpleJD()
tr = jd.sample()
lp = jd.log_prob(tr)

hmc = HMC(jd, 1E-2, 100, 200)
hmc.run()
