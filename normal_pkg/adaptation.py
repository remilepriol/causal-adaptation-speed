from collections import defaultdict

import numpy as np
import torch
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


class CholeskyModule(nn.Module):

    def __init__(self, za, la, linear, bias, lcond, BtoA=False):
        super().__init__()
        self.za = nn.Parameter(pamper(za))
        self.la = nn.Parameter(pamper(la))
        self.linear = nn.Linear()
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
        marginal = .5 * torch.sum((a @ self.la - self.za) ** 2)
        zcond = torch.linear(a)
        conditional = .5 * torch.sum((b @ self.lcond - self.zcond) ** 2)

        return marginal + conditional

    def kullback_leibler(self, other):
        """Complicated formula. Double check by going through the joint representation."""
        dim = self.za.shape[0]
        va = torch.triangular_solve(other.la, self.la, upper=False)
        marginal = .5 * torch.sum((va.t() @ self.za - other.za) ** 2)
        marginal += .5 * (torch.sum(va ** 2) - dim)
        marginal -= torch.sum(torch.log(torch.diag(va)))

        vcond = torch.triangular_solve(other.lcond, self.lcond, upper=False)
        conditional = .5 * (torch.sum(vcond ** 2) - dim)
        conditional -= torch.sum(torch.log(torch.diag(vcond)))

        mua = torch.triangular_solve(self.za, self.la.t(), upper=True)
        mat = vcond.t() @ self.linear.weight - other.linear.weight
        vec = vcond.t() @ self.linear.bias - other.linear.bias
        conditional += torch.sum((mat @ mua + vec) ** 2)
        tmp = torch.triangular_solve(mat.t(), self.la)
        conditional += torch.sum(tmp ** 2)

        return marginal + conditional


# update use torch.tril
# implement stochastic prox gradient optimizer ?
# note that's only for the cholesky matrices

class AdaptationExperiment:

    def __init__(self, k, n, T, intervention, init,
                 lr, batch_size=10, scheduler_exponent=0,  # optimizer
                 log_interval=10):
        self.k = k
        self.n = n
        self.intervention = intervention
        self.init = init
        self.lr = lr
        self.batch_size = batch_size
        self.scheduler_exponent = scheduler_exponent
        self.log_interval = 10

        reference = normal.sample(k, init)
        self.sampler = reference.to_joint().to_mean()


def experiment_optimize(k, n, T, intervention, init,
                        lr, batch_size=10, scheduler_exponent=0,  # optimizer
                        log_interval=10):
    """Record adaptation speed

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Use SGD to update a causal
    and an anticausal model for T steps. At each step, measure KL
    and distance in scores for causal and anticausal directions.
    """
    causalstatic = 0
    transferstatic = causalstatic.intervention(on=intervention, concentration=concentration)
    causal = causalstatic.to_module()
    transfer = transferstatic.to_module()

    anticausal = causalstatic.reverse().to_module()
    antitransfer = transferstatic.reverse().to_module()

    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0,
                 'weight_decay': 0}
    causaloptimizer = optim.ASGD(causal.parameters(), **optkwargs)
    antioptimizer = optim.ASGD(anticausal.parameters(), **optkwargs)
    optimizers = [causaloptimizer, antioptimizer, jointoptimizer]

    steps = []
    ans = defaultdict(list)
    for t in tqdm.tqdm(range(T)):

        # EVALUATION
        if t % log_interval == 0:
            steps.append(t)

            with torch.no_grad():

                for model, optimizer, target, name in zip(
                        [causal, anticausal, joint],
                        optimizers,
                        [transfer, antitransfer, jointtransfer],
                        ['causal', 'anti', 'joint']
                ):
                    # SGD
                    ans[f'kl_{name}'].append(target.kullback_leibler(model))
                    ans[f'scoredist_{name}'].append(target.scoredist(model))

                    # ASGD
                    with AveragedModel(model, optimizer) as m:
                        ans[f'kl_{name}_average'].append(
                            target.kullback_leibler(m))
                        ans[f'scoredist_{name}_average'].append(
                            target.scoredist(m))
        # UPDATE
        for opt in optimizers:
            opt.lr = lr / t ** scheduler_exponent
            opt.zero_grad()

        if batch_size == 'full':
            causalloss = transfer.kullback_leibler(causal).sum()
            antiloss = antitransfer.kullback_leibler(anticausal).sum()
        else:
            aa, bb = transferstatic.sample(m=batch_size, return_tensor=True)
            causalloss = - causal(aa, bb).sum() / batch_size
            antiloss = - anticausal(aa, bb).sum() / batch_size

        for loss, opt in zip([causalloss, antiloss, jointloss], optimizers):
            loss.backward()
            opt.step()

    for key, item in ans.items():
        ans[key] = torch.stack(item).numpy()

    return {'steps': np.array(steps), **ans}


def test_experiment_optimize():
    for intervention in ['cause', 'effect', 'gmechanism']:
        experiment_optimize(
            k=2, n=3, T=6, lr=.1, batch_size=4, log_interval=1,
            intervention=intervention
        )
