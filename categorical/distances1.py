from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim

from categorical.utils import kullback_leibler, logit2proba, logsumexp, proba2logit
from averaging_manager import AveragedModel


def joint2conditional(joint):
    marginal = np.sum(joint, axis=-1)
    conditional = joint / np.expand_dims(marginal, axis=-1)

    return CategoricalStatic(marginal, conditional)


def jointlogit2conditional(joint):
    sa = logsumexp(joint)
    sa -= sa.mean(axis=1, keepdims=True)
    sba = joint - sa[:, :, np.newaxis]
    sba -= sba.mean(axis=2, keepdims=True)

    return CategoricalStatic(sa, sba, from_probas=False)


def sample_joint(k, n, concentration, symmetric=True):
    """Sample n causal mechanisms of categorical variables of dimension K."""
    if symmetric:
        joint = np.random.dirichlet(concentration * np.ones(k ** 2),
                                    size=n).reshape((n, k, k))
        return joint2conditional(joint)
    else:
        pa = np.random.dirichlet(concentration * np.ones(k), size=n)
        pba = np.random.dirichlet(concentration * np.ones(k), size=[n, k])
        return CategoricalStatic(pa, pba)


class CategoricalStatic:
    """Represent n categorical distributions of variables (a,b) of dimension k each."""

    def __init__(self, marginal, conditional, from_probas=True):
        """The distribution is represented by a marginal p(a) and a conditional p(b|a)

        marginal is n*k array.
        conditional is n*k*k array. Each element conditional[i,j,k] is p_i(b=k |a=j)
        """
        self.n, self.k = marginal.shape

        if not conditional.shape == (self.n, self.k, self.k):
            raise ValueError(
                f'Marginal shape {marginal.shape} and conditional '
                f'shape {conditional.shape} do not match.')

        if from_probas:
            self.marginal = marginal
            self.conditional = conditional
            self.sa = proba2logit(marginal)
            self.sba = proba2logit(conditional)
        else:
            self.marginal = logit2proba(marginal)
            self.conditional = logit2proba(conditional)
            self.sa = marginal
            self.sba = conditional

    def to_joint(self, return_probas=True):
        if return_probas:
            return self.conditional * self.marginal[:, :, np.newaxis]
        else:  # return logits
            joint = self.sba \
                    + (self.sa - logsumexp(self.sba))[:, :, np.newaxis]
            return joint - np.mean(joint, axis=(1, 2), keepdims=True)

    def reverse(self):
        """Return conditional from b to a.
        Compute marginal pb and conditional pab such that pab*pb = pba*pa.
        """
        joint = self.to_joint(return_probas=False)
        joint = np.swapaxes(joint, 1, 2)  # invert variables
        return jointlogit2conditional(joint)

    def probadist(self, other):
        pd = np.sum((self.marginal - other.marginal) ** 2, axis=1)
        pd += np.sum((self.conditional - other.conditional) ** 2, axis=(1, 2))
        return pd

    def scoredist(self, other):
        sd = np.sum((self.sa - other.sa) ** 2, axis=1)
        sd += np.sum((self.sba - other.sba) ** 2, axis=(1, 2))
        return sd

    def sqdistance(self, other):
        """Return the squared euclidean distance between self and other"""
        return self.probadist(other), self.scoredist(other)

    def kullback_leibler(self, other):
        p0 = self.to_joint().reshape(self.n, self.k ** 2)
        p1 = other.to_joint().reshape(self.n, self.k ** 2)
        return kullback_leibler(p0, p1)

    def newmarginal(self, concentration, fromjoint):
        if fromjoint:
            return sample_joint(self.k, self.n, concentration,
                                symmetric=True).marginal
        else:
            return np.random.dirichlet(
                concentration * np.ones(self.k), size=self.n)

    def intervention(self, on, concentration, fromjoint=True):
        # sample new marginal
        if on == 'independent':
            # make cause and effect independent,
            # but without changing the effect marginal.
            newmarginal = self.reverse().marginal
        elif on == 'geometric':
            newmarginal = self.sba.mean(axis=1)
        else:
            newmarginal = self.newmarginal(concentration, fromjoint)

        # replace the cause or the effect by this marginal
        if on == 'cause':
            return CategoricalStatic(newmarginal, self.conditional)
        elif on in ['effect', 'independent', 'geometric']:  # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            return CategoricalStatic(self.marginal, newconditional)
        else:
            raise ValueError(f'Intervention on {on} is not supported.')

    def sample(self, m, return_tensor=False):
        """For each of the n distributions, return m samples. (n*m*2 array) """
        flatjoints = self.to_joint().reshape((self.n, self.k ** 2))
        samples = np.array(
            [np.random.choice(self.k ** 2, size=m, p=p) for p in flatjoints])
        a = samples // self.k
        b = samples % self.k
        if not return_tensor:
            return a, b
        else:
            return torch.from_numpy(a), torch.from_numpy(b)

    def to_module(self):
        return CategoricalModule(self.sa, self.sba)

    def __repr__(self):
        return (f"n={self.n} categorical of dimension k={self.k}\n"
                f"{self.marginal}\n"
                f"{self.conditional}")


def test_ConditionalStatic():
    print('test categorical static')

    # test the reversion formula on a known example
    pa = np.array([[.5, .5]])
    pba = np.array([[[.5, .5], [1 / 3, 2 / 3]]])
    anspb = np.array([[5 / 12, 7 / 12]])
    anspab = np.array([[[3 / 5, 2 / 5], [3 / 7, 4 / 7]]])

    test = CategoricalStatic(pa, pba).reverse()
    answer = CategoricalStatic(anspb, anspab)

    probadist, scoredist = test.sqdistance(answer)
    assert probadist < 1e-4, probadist
    assert scoredist < 1e-4, scoredist

    # ensure that reverse is reversible
    distrib = sample_joint(3, 17, 1, False)
    assert np.allclose(0, distrib.reverse().reverse().sqdistance(distrib))

    distrib.kullback_leibler(distrib.reverse())
    a, b = distrib.sample(10000)
    c = a * distrib.k + b
    approx = np.bincount(c[0]) / c.shape[1]
    joint = distrib.to_joint()[0].flatten()
    assert np.allclose(joint, approx, atol=1e-2, rtol=1e-1)


def experiment(k, n, concentration, intervention,
               symmetric_init=True, symmetric_intervention=False):
    """Sample n mechanisms of order k and for each of them sample an intervention on the desired
    mechanism. Return the distance between the original distribution and the intervened
    distribution in the causal parameter space and in the anticausal parameter space.

    :param intervention: takes value 'cause' or 'effect'
    :param symmetric_init: sample the causal parameters such that the marginals on a and b are
    drawn from the same distribution
    :param symmetric_intervention: sample the intervention parameters from the same law as the
    initial parameters
    """
    # causal parameters
    causal = sample_joint(k, n, concentration, symmetric_init)
    transfer = causal.intervention(
        on=intervention,
        concentration=concentration,
        fromjoint=symmetric_intervention
    )
    cpd, csd = causal.sqdistance(transfer)

    # anticausal parameters
    anticausal = causal.reverse()
    antitransfer = transfer.reverse()
    apd, asd = anticausal.sqdistance(antitransfer)

    return np.array([[cpd, csd], [apd, asd]])


def test_experiment():
    print('test experiment')
    for intervention in ['cause', 'effect', 'independent', 'geometric']:
        for symmetric_init in [True, False]:
            for symmetric_intervention in [True, False]:
                experiment(2, 3, 1, intervention, symmetric_init,
                           symmetric_intervention)


class CategoricalModule(nn.Module):
    """Represent n categorical conditionals as a pytorch module"""

    def __init__(self, sa, sba):
        super(CategoricalModule, self).__init__()
        self.n, self.k = tuple(sa.shape)
        self.sa = nn.Parameter(torch.tensor(sa))
        self.sba = nn.Parameter(torch.tensor(sba))

    def forward(self, a, b):
        """
        :param a: n*m collection of m class in {1,..., k} observed
        for each of the n models
        :param b: n*m like a
        :return: the log-probability of observing a,b,
        where model 1 explains first row of a,b,
        model 2 explains row 2 and so forth.
        """
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, a.shape[1])
        return self.to_joint()[rows.view(-1), a.view(-1), b.view(-1)]

    def to_joint(self):
        return F.log_softmax(self.sba, dim=2) \
               + F.log_softmax(self.sa, dim=1).unsqueeze(dim=2)

    def to_static(self):
        return CategoricalStatic(
            logit2proba(self.sa.detach().numpy()),
            logit2proba(self.sba.detach().numpy())
        )

    def kullback_leibler(self, other):
        joint = self.to_joint()
        return torch.sum((joint - other.to_joint()) * torch.exp(joint),
                         dim=(1, 2))

    def scoredist(self, other):
        return torch.sum((self.sa - other.sa) ** 2, dim=1) \
               + torch.sum((self.sba - other.sba) ** 2, dim=(1, 2))

    def __repr__(self):
        return f"CategoricalModule(joint={self.to_joint().detach()})"


def test_CategoricalModule(n=7, k=5):
    print('test categorical module')
    references = sample_joint(k, n, 1)
    intervened = references.intervention(on='cause', concentration=1)

    modules = references.to_module()

    # test that reverse is numerically stable
    kls = references.reverse().reverse().to_module().kullback_leibler(modules)
    assert torch.allclose(torch.zeros(n, dtype=torch.double), kls), kls

    # test optimization
    optimizer = optim.SGD(modules.parameters(), lr=1)
    aa, bb = intervened.sample(13, return_tensor=True)
    negativeloglikelihoods = -modules(aa, bb).mean()
    optimizer.zero_grad()
    negativeloglikelihoods.backward()
    optimizer.step()

    imodules = intervened.to_module()
    imodules.kullback_leibler(modules)
    imodules.scoredist(modules)


class JointModule(nn.Module):

    def __init__(self, logits):
        super(JointModule, self).__init__()
        self.n, k2 = logits.shape  # logits is flat

        self.k = int(np.sqrt(k2))
        if self.k ** 2 != k2:
            raise ValueError('logits matrix can not be reshaped to square.')

        self.logits = nn.Parameter(logits)

    def forward(self, a, b):
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, a.shape[1]).view(-1)
        index = (a * self.k + b).view(-1)
        return F.log_softmax(self.logits, dim=1)[rows, index]

    def kullback_leibler(self, other):
        return torch.sum((self.logits - other.logits) * F.softmax(self.logits, dim=1), dim=1)

    def scoredist(self, other):
        return torch.sum((self.logits - other.logits) ** 2, dim=1)

    def __repr__(self):
        return f"CategoricalJoint(logits={self.logits.detach()})"


class MaximumLikelihoodEstimator:

    def __init__(self, n, k):
        self.counts = torch.ones((n, k, k), dtype=torch.double)

    @property
    def total(self):
        return self.counts.sum(dim=(1, 2))

    @property
    def frequencies(self):
        return self.counts / self.total.unsqueeze(1).unsqueeze(2)

    def update(self, a, b):
        rows = torch.arange(0, len(a)).unsqueeze(1).repeat(1, a.shape[1])
        self.counts[rows.view(-1), a.view(-1), b.view(-1)] += 1

    def to_joint(self):
        return torch.log(self.frequencies)


def init_mle(n0: int, static: CategoricalStatic):
    mle = MaximumLikelihoodEstimator(static.n, static.k)
    mle.counts = n0 * torch.from_numpy(static.to_joint(return_probas=True))
    return mle


def experiment_optimize(k, n, T, lr, concentration, intervention,
                        is_init_symmetric=True,
                        is_intervention_symmetric=False,
                        batch_size=10, scheduler_exponent=0, n0=10,
                        log_interval=10):
    """Measure optimization speed and parameters distance.

    Hypothesis: initial distance to optimum is correlated to optimization speed with SGD.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Use SGD to update a causal
    and an anticausal model for T steps. At each step, measure KL
    and distance in scores for causal and anticausal directions.

    :param intervention: takes value 'cause' or 'effect'
    :param is_init_symmetric: sample the causal parameters such that the marginals on a and b are
    drawn from the same distribution
    :param is_intervention_symmetric: sample the intervention parameters from the same law as the
    initial parameters
    """
    causalstatic = sample_joint(k, n, concentration, is_init_symmetric)
    transferstatic = causalstatic.intervention(
        on=intervention,
        concentration=concentration,
        fromjoint=is_intervention_symmetric
    )
    causal = causalstatic.to_module()
    transfer = transferstatic.to_module()

    anticausal = causalstatic.reverse().to_module()
    antitransfer = transferstatic.reverse().to_module()

    joint = JointModule(causal.to_joint().detach().view(n, -1))
    jointtransfer = JointModule(transfer.to_joint().detach().view(n, -1))

    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0,
                 'weight_decay': 0}
    causaloptimizer = optim.ASGD(causal.parameters(), **optkwargs)
    antioptimizer = optim.ASGD(anticausal.parameters(), **optkwargs)
    jointoptimizer = optim.ASGD(joint.parameters(), **optkwargs)
    optimizers = [causaloptimizer, antioptimizer, jointoptimizer]

    # MAP
    countestimator = MaximumLikelihoodEstimator(n, k)
    priorestimator = init_mle(n0, causalstatic)

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

                # MAP
                ans['kl_MAP_uniform'].append(
                    transfer.kullback_leibler(countestimator))
                ans['kl_MAP_source'].append(
                    transfer.kullback_leibler(priorestimator))

        # UPDATE
        for opt in optimizers:
            opt.lr = lr / t ** scheduler_exponent
            opt.zero_grad()

        if batch_size == 'full':
            causalloss = transfer.kullback_leibler(causal).sum()
            antiloss = antitransfer.kullback_leibler(anticausal).sum()
            jointloss = jointtransfer.kullback_leibler(joint).sum()
        else:
            aa, bb = transferstatic.sample(m=batch_size, return_tensor=True)
            causalloss = - causal(aa, bb).sum() / batch_size
            antiloss = - anticausal(bb, aa).sum() / batch_size
            jointloss = - joint(aa, bb).sum() / batch_size
            countestimator.update(aa, bb)
            priorestimator.update(aa, bb)

        for loss, opt in zip([causalloss, antiloss, jointloss], optimizers):
            loss.backward()
            opt.step()

    for key, item in ans.items():
        ans[key] = torch.stack(item).numpy()

    return {'steps': np.array(steps), **ans}


def test_experiment_optimize():
    experiment_optimize(k=2, n=3, T=6, lr=.1,
                        batch_size=4,
                        log_interval=1,
                        concentration=1,
                        intervention='cause')


if __name__ == "__main__":
    test_ConditionalStatic()
    test_experiment()
    test_CategoricalModule()
    test_experiment_optimize()
