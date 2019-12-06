from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy import stats
from torch import nn, optim

from averaging_manager import AveragedModel
from categorical.utils import kullback_leibler, logit2proba, logsumexp, proba2logit


def joint2conditional(joint):
    marginal = np.sum(joint, axis=-1)
    conditional = joint / np.expand_dims(marginal, axis=-1)

    return CategoricalStatic(marginal, conditional)


def jointlogit2conditional(joint, is_btoa):
    sa = logsumexp(joint)
    sa -= sa.mean(axis=1, keepdims=True)
    sba = joint - sa[:, :, np.newaxis]
    sba -= sba.mean(axis=2, keepdims=True)

    return CategoricalStatic(sa, sba, from_probas=False, is_btoa=is_btoa)


def sample_joint(k, n, concentration=1, dense=False, logits=True):
    """Sample n causal mechanisms of categorical variables of dimension K.

    The concentration argument specifies the concentration of the resulting cause marginal.
    """
    if logits:
        sa = stats.loggamma.rvs(concentration, size=(n, k))
        if dense:
            sba = stats.loggamma.rvs(concentration, size=(n, k, k))
        else:
            # use the fact that a loggamma is well approximated by a negative exponential
            # for small values of the shape parameter with the transformation scale = 1/ shape
            sba = - stats.expon.rvs(scale=k / concentration, size=(n, k, k))
        sa -= sa.mean(axis=1, keepdims=True)
        sba -= sba.mean(axis=2, keepdims=True)
        return CategoricalStatic(sa, sba, from_probas=False)
    else:
        pa = np.random.dirichlet(concentration * np.ones(k), size=n)
        condconcentration = concentration if dense else concentration / k
        pba = np.random.dirichlet(condconcentration * np.ones(k), size=[n, k])
        return CategoricalStatic(pa, pba, from_probas=True)


class CategoricalStatic:
    """Represent n categorical distributions of variables (a,b) of dimension k each."""

    def __init__(self, marginal, conditional, from_probas=True, is_btoa=False):
        """The distribution is represented by a marginal p(a) and a conditional p(b|a)

        marginal is n*k array.
        conditional is n*k*k array. Each element conditional[i,j,k] is p_i(b=k |a=j)
        """
        self.n, self.k = marginal.shape
        self.BtoA = is_btoa

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
        return jointlogit2conditional(joint, not self.BtoA)

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

    def intervention(self, on, concentration=1):
        # sample new marginal
        if on == 'independent':
            # make cause and effect independent,
            # but without changing the effect marginal.
            newmarginal = self.reverse().marginal
        elif on == 'geometric':
            newmarginal = logit2proba(self.sba.mean(axis=1))
        elif on == 'weightedgeo':
            newmarginal = logit2proba(np.sum(self.sba * self.marginal[:, :, None], axis=1))
        else:
            newmarginal = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)

        # TODO use logits of the marginal for stability certainty
        # replace the cause or the effect by this marginal
        if on == 'cause':
            return CategoricalStatic(newmarginal, self.conditional)
        elif on in ['effect', 'independent', 'geometric', 'weightedgeo']:
            # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            return CategoricalStatic(self.marginal, newconditional)
        elif on == 'mechanism':
            # TODO THIS IS WRONG AND UNSTABLE
            # sample from a dirichlet centered on each conditional
            sba = np.zeros_like(self.sba)  # in logits space
            for i in range(self.n):
                for j in range(self.k):
                    for k in range(self.k):
                        p = self.conditional[i, j, k]
                        if p < 1e-10:  # small
                            sba[i, j, k] = - stats.expon.rvs(scale=1 / p)
                        else:
                            sba[i, j, k] = stats.loggamma.rvs(p)
            sba -= sba.mean(axis=2, keepdims=True)
            return CategoricalStatic(self.sa, sba, from_probas=False)
        elif on == 'gmechanism':
            # sample from a gaussian centered on each conditional
            sba = np.random.normal(self.sba, self.sba.std())
            sba -= sba.mean(axis=2, keepdims=True)
            return CategoricalStatic(self.sa, sba, from_probas=False)
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
        return CategoricalModule(self.sa, self.sba, is_btoa=self.BtoA)

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
    distrib = sample_joint(3, 17, 1, True)
    assert np.allclose(0, distrib.reverse().reverse().sqdistance(distrib))

    distrib.kullback_leibler(distrib.reverse())
    n = 10000
    a, b = distrib.sample(n)
    c = a * distrib.k + b
    val, approx = np.unique(c[0], return_counts=True)
    approx = approx.astype(float) / n
    joint = distrib.to_joint()[0].flatten()
    assert np.allclose(joint, approx, atol=1e-2, rtol=1e-1), print(joint, approx)


def experiment(k, n, concentration, intervention, dense_init=True):
    """Sample n mechanisms of order k and for each of them sample an intervention on the desired
    mechanism. Return the distance between the original distribution and the intervened
    distribution in the causal parameter space and in the anticausal parameter space.
    """
    # causal parameters
    causal = sample_joint(k, n, concentration, dense_init)
    transfer = causal.intervention(on=intervention, concentration=concentration)
    cpd, csd = causal.sqdistance(transfer)

    # anticausal parameters
    anticausal = causal.reverse()
    antitransfer = transfer.reverse()
    apd, asd = anticausal.sqdistance(antitransfer)

    return np.array([[cpd, csd], [apd, asd]])


def test_experiment():
    print('test experiment')
    for intervention in ['cause', 'effect', 'mechanism', 'gmechanism', 'independent', 'geometric',
                         'weightedgeo']:
        for dense_init in [True, False]:
            experiment(2, 3, 1, intervention, dense_init)


class CategoricalModule(nn.Module):
    """Represent n categorical conditionals as a pytorch module"""

    def __init__(self, sa, sba, is_btoa=False):
        super(CategoricalModule, self).__init__()
        self.n, self.k = tuple(sa.shape)

        sa = sa.clone().detach() if torch.is_tensor(sa) else  torch.tensor(sa)
        sba = sba.clone().detach() if torch.is_tensor(sba) else torch.tensor(sba)
        self.sa = nn.Parameter(sa.to(torch.float32))
        self.sba = nn.Parameter(sba.to(torch.float32))
        self.BtoA = is_btoa

    def forward(self, a, b):
        """
        :param a: n*m collection of m class in {1,..., k} observed
        for each of the n models
        :param b: n*m like a
        :return: the log-probability of observing a,b,
        where model 1 explains first row of a,b,
        model 2 explains row 2 and so forth.
        """
        if self.BtoA:
            a, b = b, a
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
    assert torch.allclose(torch.zeros(n), kls), kls

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
            raise ValueError('Logits matrix can not be reshaped to square.')

        # normalize to sum to 0
        logits = logits - logits.mean(dim=1, keepdim=True)
        self.logits = nn.Parameter(logits)

    @property
    def logpartition(self):
        return torch.logsumexp(self.logits, dim=1)

    def forward(self, a, b):
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, a.shape[1]).view(-1)
        index = (a * self.k + b).view(-1)
        return F.log_softmax(self.logits, dim=1)[rows, index]

    def kullback_leibler(self, other):
        if isinstance(other, CategoricalModule):
            ologits = other.to_joint().detach().view(self.n, self.k**2)
            print(torch.logsumexp(ologits, dim=1))
        elif isinstance(other, JointModule):
            ologits = other.logits
        else:
            raise ValueError(other)
        a = self.logpartition
        kl = torch.sum((self.logits - ologits) * torch.exp(self.logits - a[:, None]), dim=1)
        return kl - a + other.logpartition

    def scoredist(self, other):
        return torch.sum((self.logits - other.logits) ** 2, dim=1)

    def __repr__(self):
        return f"CategoricalJoint(logits={self.logits.detach()})"


class MaximumLikelihoodEstimator:

    def __init__(self, n, k):
        self.counts = torch.ones((n, k, k))

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


def experiment_optimize(k, n, T, lr, intervention,
                        concentration=1,
                        is_init_dense=False,
                        batch_size=10, scheduler_exponent=0, n0=10,
                        log_interval=10, use_map=False):
    """Measure optimization speed and parameters distance.

    Hypothesis: initial distance to optimum is correlated to optimization speed with SGD.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Use SGD to update a causal
    and an anticausal model for T steps. At each step, measure KL
    and distance in scores for causal and anticausal directions.
    """
    causalstatic = sample_joint(k, n, concentration, is_init_dense)
    transferstatic = causalstatic.intervention(on=intervention, concentration=concentration)
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
                if use_map:
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
            antiloss = - anticausal(aa, bb).sum() / batch_size
            jointloss = - joint(aa, bb).sum() / batch_size
            if use_map:
                countestimator.update(aa, bb)
                priorestimator.update(aa, bb)

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


def experiment_guess(
        k, n, T, lr, intervention,
        concentration=1, is_init_dense=False,
        batch_size=10, scheduler_exponent=0, log_interval=10
):
    """Measure optimization speed after guessing intervention.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Initialize a causal and
    an anticausal model to the reference distribution. Duplicate them
    and initialize one module of each duplicata to the uniform.
    Run the optimization and record KL and distance. Also record the
    accuracy of guessing the intervention based on the lowest KL.
    """
    causalstatic = sample_joint(k, n, concentration, is_init_dense)
    transferstatic = causalstatic.intervention(on=intervention, concentration=concentration)
    causal = causalstatic.to_module()
    transfer = transferstatic.to_module()

    anticausal = causalstatic.reverse().to_module()
    antitransfer = transferstatic.reverse().to_module()

    joint = JointModule(causal.to_joint().detach().view(n, -1))
    jointtransfer = JointModule(transfer.to_joint().detach().view(n, -1))

    # TODO put all models and their transfer into a dict
    models = [causal, anticausal, joint]
    targets = [transfer, antitransfer, jointtransfer]

    # step 1 : duplicate models with intervention guessing
    causalguessA = CategoricalModule(torch.zeros([n, k]), causal.sba)
    causalguessB = CategoricalModule(causal.sa, torch.zeros([n, k, k]))

    antiguessB = CategoricalModule(torch.zeros([n, k]), anticausal.sba)
    antiguessA = CategoricalModule(anticausal.sa, torch.zeros([n, k, k]))

    models += [causalguessA, causalguessB, antiguessA, antiguessB]
    targets += [transfer, transfer, antitransfer, antitransfer]

    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0, 'weight_decay': 0}
    optimizer = optim.ASGD([p for m in models for p in m.parameters()], **optkwargs)

    steps = []
    ans = defaultdict(list)
    for step in tqdm.tqdm(range(T)):

        # EVALUATION
        if step % log_interval == 0:
            steps.append(step)
            with torch.no_grad():

                for model, target, name in zip(models, targets, ['causal', 'anti', 'joint']):
                    # SGD
                    ans[f'kl_{name}'].append(target.kullback_leibler(model))
                    ans[f'scoredist_{name}'].append(target.scoredist(model))

                    # ASGD
                    with AveragedModel(model, optimizer):
                        ans[f'kl_{name}_average'].append(target.kullback_leibler(model))
                        ans[f'scoredist_{name}_average'].append(target.scoredist(model))

        # UPDATE
        optimizer.lr = lr / step ** scheduler_exponent
        optimizer.zero_grad()

        if batch_size == 'full':
            loss = sum([t.kullback_leibler(m).sum() for m, t in zip(models, targets)])
        else:
            aa, bb = transferstatic.sample(m=batch_size, return_tensor=True)
            loss = sum([- m(aa, bb).sum() for m in models]) / batch_size

            # step 2, estimate likelihood of samples aa and bb for each marginals
            # of the reference model and take the lowest likelihood as a guess
            # for the intervention. Take the average over all examples seen until now

        loss.backward()
        optimizer.step()

    for key, item in ans.items():
        ans[key] = torch.stack(item).numpy()

    return {'steps': np.array(steps), **ans}


def test_experiment_guess():
    for bs in ['full', 4]:
        for intervention in ['cause', 'effect']:
            experiment_guess(
                k=5, n=10, T=6, lr=.1, batch_size=bs, log_interval=1,
                intervention=intervention
            )


if __name__ == "__main__":
    test_ConditionalStatic()
    test_experiment()
    test_CategoricalModule()
    test_experiment_optimize()
    test_experiment_guess()
