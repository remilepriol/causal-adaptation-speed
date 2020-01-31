import copy
from collections import defaultdict

import numpy as np
import torch
import tqdm
from torch import nn, optim

from averaging_manager import AveragedModel
from normal_pkg import normal


def pamper(a):
    if torch.is_tensor(a):
        b = a.clone().detach()
    elif isinstance(a, np.ndarray):
        b = torch.from_numpy(a)
    else:
        b = torch.tensor(a)
    return b.to(torch.float32)


def cholesky_numpy2module(cho, BtoA):
    return CholeskyModule(cho.za, cho.la, cho.linear, cho.bias, cho.lcond, BtoA)


class CholeskyModule(nn.Module):

    def __init__(self, za, la, linear, bias, lcond, BtoA=False):
        super().__init__()
        dim = za.shape[0]
        self.za = nn.Parameter(pamper(za))
        self.la = nn.Parameter(pamper(la))
        self.linear = nn.Linear(dim, dim)
        self.linear.weight.data = pamper(linear)
        self.linear.bias.data = pamper(bias)
        self.lcond = pamper(lcond)
        self.BtoA = BtoA

    def forward(self, a, b):
        """Compute only the quadratic part of the loss to get its gradient.
        I will use a proximal operator to optimize the log-partition logdet-barrier.
        """
        batch_size = a.shape[0]
        if self.BtoA:
            a, b = b, a
        marginal = .5 * torch.sum((a @ self.la - self.za) ** 2) / batch_size
        zcond = self.linear(a)
        conditional = .5 * torch.sum((b @ self.lcond - zcond) ** 2) / batch_size

        # for now do plain gradient descent !
        marginal -= torch.sum(torch.log(torch.diag(self.la)))
        conditional -= torch.sum(torch.log(torch.diag(self.lcond)))

        return marginal + conditional

    def kullback_leibler(self, other):
        """Complicated formula. Double check by going through the joint representation."""
        dim = self.za.shape[0]
        va, _ = torch.trtrs(other.la, self.la, upper=False)
        marginal = .5 * torch.sum((va.t() @ self.za - other.za) ** 2)
        marginal += .5 * (torch.sum(va ** 2) - dim)
        marginal -= torch.sum(torch.log(torch.diag(va)))

        vcond, _ = torch.trtrs(other.lcond, self.lcond, upper=False)
        conditional = .5 * (torch.sum(vcond ** 2) - dim)
        conditional -= torch.sum(torch.log(torch.diag(vcond)))

        mua, _ = torch.trtrs(self.za, self.la.t(), upper=True)
        mat = vcond.t() @ self.linear.weight - other.linear.weight
        vec = vcond.t() @ self.linear.bias - other.linear.bias
        conditional += torch.sum((mat @ mua + vec) ** 2)
        tmp, _ = torch.trtrs(mat.t(), self.la)
        conditional += torch.sum(tmp ** 2)

        return marginal + conditional


# update use torch.tril
# implement stochastic prox gradient optimizer ?
# note that's only for the cholesky matrices

class AdaptationExperiment:
    """Record adaptation speed."""

    def __init__(self, k, n, intervention, init,
                 lr, batch_size=10, scheduler_exponent=0,  # optimizer
                 log_interval=10):
        self.k = k
        self.n = n
        self.intervention = intervention
        self.init = init
        self.lr = lr
        self.batch_size = batch_size
        self.scheduler_exponent = scheduler_exponent
        self.log_interval = log_interval

        reference = normal.sample(k, init)
        self.sampler = reference.to_mean()

        self.models = {
            'causal': cholesky_numpy2module(reference.to_cholesky(), BtoA=False),
            'anti': cholesky_numpy2module(reference.reverse().to_cholesky(), BtoA=True)
        }

        self.targets = copy.deepcopy(self.models)

        optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0, 'weight_decay': 0}
        self.optimizer = optim.ASGD(
            [p for m in self.models.values() for p in m.parameters()], **optkwargs)

        self.trajectory = defaultdict(list)
        self.t = 0

    def evaluate(self):
        # EVALUATION
        self.trajectory['steps'].append(torch.tensor(self.t))
        with torch.no_grad():
            for name, model in self.models.items():
                target = self.targets[name]
                # SGD
                self.trajectory[f'kl_{name}'].append(
                    target.kullback_leibler(model))
                # ASGD
                with AveragedModel(model, self.optimizer):
                    self.trajectory[f'kl_{name}_average'].append(
                        target.kullback_leibler(model))

    def step(self):
        self.t += 1
        self.optimizer.lr = self.lr / self.t ** self.scheduler_exponent
        self.optimizer.zero_grad()

        if self.batch_size == 'full':
            loss = sum([self.targets[name].kullback_leibler(model)
                        for name, model in self.models.items()])
        else:
            aa, bb = self.sampler.sample(self.batch_size)
            loss = sum([model(pamper(aa), pamper(bb))
                        for model in self.models.values()])
        loss.backward()
        self.optimizer.step()

    def run(self, T):
        for t in tqdm.tqdm(range(T)):
            if t % self.log_interval == 0:
                self.evaluate()
            self.step()

        for key, item in self.trajectory.items():
            self.trajectory[key] = torch.stack(item).numpy()

        return self.trajectory


def test_AdaptationExperiment():
    for intervention in ['cause', 'effect']:
        for init in ['cholesky', 'natural']:
            exp = AdaptationExperiment(
                k=2, n=3, lr=.1, batch_size=4, log_interval=1,
                intervention=intervention, init=init
            )
            exp.run(5)


if __name__ == "__main__":
    test_AdaptationExperiment()
