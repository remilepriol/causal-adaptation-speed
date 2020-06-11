import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import tqdm
from torch import nn

from averaging_manager import AveragedModel
from normal_pkg import normal
from normal_pkg.proximal_optimizer import PerturbedProximalGradient


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
        self.linear = nn.Linear(dim, dim, bias=True)
        self.linear.weight.data = pamper(linear)
        self.linear.bias.data = pamper(bias)
        self.lcond = nn.Parameter(pamper(lcond))
        self.BtoA = BtoA

        self.la.triangular = True
        self.lcond.triangular = True

    def forward(self, a, b, test_via_joint=False, nograd_logdet=False):
        """Compute only the quadratic part of the loss to get its gradient.
        I will use a proximal operator to optimize the log-partition logdet-barrier.
        """
        batch_size = a.shape[0]
        if self.BtoA:
            a, b = b, a

        # use conditional parametrization
        marginal = .5 * torch.sum((a @ self.la - self.za) ** 2) / batch_size
        zcond = self.linear(a)
        conditional = .5 * torch.sum((b @ self.lcond - zcond) ** 2) / batch_size

        logdet = - torch.sum(torch.log(torch.diag(self.la)))
        logdet += - torch.sum(torch.log(torch.diag(self.lcond)))
        if nograd_logdet:
            logdet.detach_()

        loss1 = marginal + conditional + logdet

        # use joint parametrization
        # CAREFUL the joint parametrization inverts the roles
        # of cause and effect for simplicity
        if test_via_joint:
            x = torch.cat([b, a], 1)
            z, L = self.joint_parameters()
            quadratic = .5 * torch.sum((x @ L - z) ** 2) / batch_size
            logdet = - torch.sum(torch.log(torch.diag(L)))
            loss2 = quadratic + logdet

            assert torch.isclose(loss1, loss2), (loss1, loss2)

        return loss1

    def joint_parameters(self):
        """Return joint cholesky, with order of X and Y inverted."""
        zeta = torch.cat([self.linear.bias, self.za])
        L = torch.cat([
            torch.cat([self.lcond, torch.zeros_like(self.lcond)], 1),
            torch.cat([- self.linear.weight.t(), self.la], 1)
        ], 0)
        return zeta, L

    def dist(self, other):
        return (
                torch.sum((self.za - other.za) ** 2)
                + torch.sum((self.la - other.la) ** 2)
                + torch.sum((self.linear.weight - other.linear.weight) ** 2)
                + torch.sum((self.linear.bias - other.linear.bias) ** 2)
                + torch.sum((self.lcond - other.lcond) ** 2)
        )

    def __repr__(self):
        return (
            f'CholeskyModule('
            f'\n \t za={self.za.data},'
            f'\n \t la={self.la.data},'
            f'\n \t linear={self.linear.weight.data},'
            f'\n \t bias={self.linear.bias.data},'
            f'\n \t lcond={self.lcond.data})'
        )


def cholesky_kl(p0: CholeskyModule, p1: CholeskyModule, decompose=False, nograd_logdet=False):
    z0, L0 = p0.joint_parameters()
    z1, L1 = p1.joint_parameters()
    V, _ = torch.triangular_solve(L1, L0, upper=False)
    diff = V @ z0 - z1
    vecnorm = .5 * torch.sum(diff ** 2)
    matnorm = .5 * (torch.sum(V ** 2) - z0.shape[0])
    logdet = - torch.sum(torch.log(torch.diag(V)))
    if nograd_logdet:
        logdet.detach_()

    matdivergence = matnorm + logdet
    total = vecnorm + matdivergence
    if decompose:
        return {'vector': vecnorm.item(), 'matrix': matdivergence.item(),
                'total': total.item(), 'v/t': vecnorm.item() / total.item(),
                'm/t': matdivergence.item() / total.item()}
    else:
        return total


class AdaptationExperiment:
    """Sample one distribution, adapt and record adaptation speed."""

    def __init__(
            self, T, log_interval,  # recording
            k, intervention, init, preccond_scale, intervention_scale,  # distributions
            lr=.1, batch_size=10, scheduler_exponent=0, use_prox=False,  # optimizer
    ):
        self.k = k
        self.intervention = intervention
        self.init = init
        self.lr = lr
        self.scheduler_exponent = scheduler_exponent
        self.log_interval = log_interval
        self.use_prox = use_prox

        reference = normal.sample(k, init, scale=preccond_scale)
        transfer = reference.intervention(intervention, intervention_scale)

        self.deterministic = True if batch_size == 0 else False
        if not self.deterministic:
            meanjoint = transfer.to_joint().to_mean()
            sampler = torch.distributions.MultivariateNormal(
                pamper(meanjoint.mean), covariance_matrix=pamper(meanjoint.cov)
            )
            data_size = torch.tensor([T, batch_size])
            self.dataset = sampler.sample(data_size)

        self.models = {
            'causal': cholesky_numpy2module(reference.to_cholesky(), BtoA=False),
            'anti': cholesky_numpy2module(reference.reverse().to_cholesky(), BtoA=True)
        }

        self.targets = {
            'causal': cholesky_numpy2module(transfer.to_cholesky(), BtoA=False),
            'anti': cholesky_numpy2module(transfer.reverse().to_cholesky(), BtoA=True)
        }

        optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0, 'weight_decay': 0}
        self.optimizer = PerturbedProximalGradient(
            [p for m in self.models.values() for p in m.parameters()], self.use_prox, **optkwargs)

        self.trajectory = defaultdict(list)
        self.step = 0

    def evaluate(self):
        self.trajectory['steps'].append(self.step)
        with torch.no_grad():
            for name, model in self.models.items():
                target = self.targets[name]
                # SGD
                kl = cholesky_kl(target, model).item()
                dist = target.dist(model).item()
                self.trajectory[f'kl_{name}'].append(kl)
                self.trajectory[f'scoredist_{name}'].append(dist)
                # ASGD
                with AveragedModel(model, self.optimizer):
                    kl = cholesky_kl(target, model).item()
                    dist = target.dist(model).item()
                    self.trajectory[f'kl_{name}_average'].append(kl)
                    self.trajectory[f'scoredist_{name}_average'].append(dist)

    def iterate(self):
        self.step += 1
        self.optimizer.lr = self.lr / self.step ** self.scheduler_exponent
        self.optimizer.zero_grad()

        if self.deterministic:
            loss = sum([cholesky_kl(self.targets[name], model, nograd_logdet=self.use_prox)
                        for name, model in self.models.items()])
        else:
            samples = self.dataset[self.step - 1]
            aa, bb = samples[:, :self.k], samples[:, self.k:]
            loss = sum(
                [model(aa, bb, nograd_logdet=self.use_prox) for model in self.models.values()])
        loss.backward()
        self.optimizer.step()
        self.trajectory['loss'].append(loss.item())

    def run(self, T):
        for t in range(T):
            if t % self.log_interval == 0:
                self.evaluate()
            self.iterate()
        # print(self.__repr__())

    def __repr__(self):
        return f'AdaptationExperiment step={self.step} \n' \
               + '\n'.join([f'{name} \t {model}' for name, model in self.models.items()])


def batch_adaptation(n, T, **parameters):
    trajectories = defaultdict(list)
    models = []
    for _ in tqdm.tqdm(range(n)):
        exp = AdaptationExperiment(T=T, **parameters)
        exp.run(T)
        for key, item in exp.trajectory.items():
            trajectories[key].append(item)
        models += [(exp.models, exp.targets)]

    for key, item in trajectories.items():
        trajectories[key] = np.array(item).T
    trajectories['steps'] = trajectories['steps'][:, 0]

    return trajectories, models


def sweep_lr(lrlr, base_experiment, seed=1, savedir='normal_results'):
    results = []
    print(base_experiment)
    for lr in lrlr:
        np.random.seed(seed)
        torch.manual_seed(seed)
        parameters = {'lr': lr, 'scheduler_exponent': 0, **base_experiment}
        trajectory, models = batch_adaptation(**parameters)
        results.append({
            'hyperparameters': parameters,
            'trajectory': trajectory,
            'models': models
        })

    os.makedirs(savedir, exist_ok=True)
    savefile = '{intervention}_{init}_k={k}.pkl'.format(**base_experiment)
    savepath = os.path.join(savedir, savefile)
    with open(savepath, 'wb') as fout:
        print("Saving results in ", savepath)
        pickle.dump(results, fout)


def test_AdaptationExperiment():
    batch_adaptation(T=100, k=3, n=1, lr=.1, batch_size=1, log_interval=10,
                     intervention='cause', init='natural')


if __name__ == "__main__":
    # test_AdaptationExperiment()

    base = {'n': 100, 'T': 400, 'batch_size': 1, 'use_prox': True,
            'log_interval': 10, 'intervention_scale': 1,
            'init': 'natural', 'preccond_scale': 10}
    lrlr = [.03]
    for k in [10]:
        base['k'] = k
        sweep_lr(lrlr, {**base, 'intervention': 'cause'})
        sweep_lr(lrlr, {**base, 'intervention': 'effect'})
        sweep_lr(lrlr, {**base, 'intervention': 'mechanism'})
