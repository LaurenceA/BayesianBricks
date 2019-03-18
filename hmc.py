import torch as t

class HMCTensor():
    def __init__ (self, vit, chain_length, batch_size=t.Size([])):
        self.size = vit.size
        self.batch_size = batch_size
        self.full_size = batch_size + self.size

        self.inv_mass = vit.variance()
        self.sqrt_mass = vit.inv_std()

        #record of samples
        self.samples = t.zeros(t.Size([chain_length]) + self.full_size, device="cpu")

        #state of Markov chain 
        self.x_mcmc = vit.sample(batch_size)

        #state of leapfrog integrator
        self.x = self.x_mcmc.clone().requires_grad_()
        self.p = t.zeros(vit.size)

        assert not self.x_mcmc.requires_grad
        assert     self.x.requires_grad
        assert not self.p.requires_grad

    def momentum_step(self, rate):
        self.p.add_(rate, self.x.grad)
        self.p.add_(-rate, self.x.data)

    def position_step(self, rate):
        self.x.data.addcmul_(rate, self.inv_mass, self.p)

    def zero_grad(self):
        if self.x.grad is not None:
            self.x.grad.fill_(0.)

    def log_prior_xp(self):
        lp_x = -0.5*(self.x**2).sum()
        lp_p = -0.5*(self.inv_mass*self.p**2).sum()
        return lp_x + lp_p

    def record_sample(self, i):
        self.samples[i,...] = self.x_mcmc

    def accept(self):
        self.x_mcmc.fill_(self.x)

    def refresh_momentum(self):
        self.p.normal_(0., 1.)
        self.p.mul_(self.sqrt_mass)
        

class HMCDict():
    def __init__(self, vi, chain_length):
        self.hmcts = {}
        for key, val in vi._modules.items():
            if isinstance(val, dict):
                result = HMCDict(val, chain_length)
            else:
                result = HMCTensor(val, chain_length)
            self.hmcts[key] = result

    def accept(self):
        for v in self.hmcts.values():
            v.accept()

    def record_sample(self, i):
        for v in self.hmcts.values():
            v.record_sample(i)

    def zero_grad(self):
        for v in self.hmcts.values():
            v.zero_grad()

    def position_step(self, rate):
        for v in self.hmcts.values():
            v.position_step(rate)

    def refresh_momentum(self):
        for v in self.hmcts.values():
            v.refresh_momentum()

    def momentum_step(self, rate):
        for v in self.hmcts.values():
            v.momentum_step(rate)

    def log_prior_xp(self):
        total = 0.
        for v in self.hmcts.values():
            total += v.log_prior_xp()
        return total

    def __getitem__(self, key):
        val = self.hmcts[key]
        if isinstance(val, HMCTensor):
            return val.x
        else:
            return val

class HMC():
    def __init__(self, fn, vi, chain_length, warmup=0, rate=1E-2, trajectory_length=1.):
        self.fn = fn
        self.rate = rate
        self.steps = int(trajectory_length // rate)
        self.chain_length = chain_length
        self.warmup = warmup

        self.tensors = HMCDict(vi.tensors, chain_length)

    def zero_grad(self):
        self.tensors.zero_grad()

    def position_step(self, rate):
        self.tensors.position_step(rate)

    def log_prior_xp(self):
        return self.tensors.log_prior_xp()

    def momentum_step(self, rate):
        self.tensors.zero_grad()
        lp = self.fn(self.tensors)
        lp.backward()
        self.tensors.momentum_step(rate)
        return lp


    def step(self, i=None):
        self.tensors.refresh_momentum()

        lp_prior_xp = self.log_prior_xp()
        lp_like     = self.momentum_step(0.5*self.rate)
        lp_init     = lp_prior_xp + lp_like

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
        if t.rand(()) < acceptance_prob:
            self.tensors.accept()
            #for v in self.hmcts.values():
            #    v.x_mcmc.fill_(v.x)

        if i is not None:
            self.tensors.record_sample(i)

    def run(self):
        for _ in range(self.warmup):
            self.step()

        for i in range(self.chain_length):
            self.step(i)
