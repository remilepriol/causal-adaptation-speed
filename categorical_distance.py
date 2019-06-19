import numpy as np
import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F


def kullback_leibler(p1, p2):
    return np.sum(p1 * np.log(p1 / p2), axis=-1)


def proba2logit(p):
    s = np.log(p)
    s -= np.mean(s, axis=-1, keepdims=True)
    return s


def logsumexp(s):
    smax = np.amax(s, axis=-1)
    return smax + np.log(np.sum(np.exp(s - np.expand_dims(smax, axis=-1)), axis=-1))


def logit2proba(s):
    return np.exp(s - np.expand_dims(logsumexp(s), axis=-1))


def test_proba2logit():
    p = np.random.dirichlet(np.ones(50), size=300)
    s = proba2logit(p)
    assert np.allclose(0, np.sum(s, axis=-1))

    q = logit2proba(s)
    assert np.allclose(1, np.sum(q, axis=-1)), q
    assert np.allclose(p, q), p - q


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
        joint = np.random.dirichlet(concentration * np.ones(k ** 2), size=n).reshape((n, k, k))
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
            raise ValueError(f'Marginal shape {marginal.shape} and conditional '
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
            joint = self.sba + (self.sa - logsumexp(self.sba))[:, :, np.newaxis]
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

    def intervention(self, on, concentration, fromjoint=True):
        # sample new marginal
        if on == 'independent':
            # make cause and effect independent,
            # but without changing the effect marginal.
            newmarginal = self.reverse().marginal
        else:
            if fromjoint:
                newmarginal = sample_joint(self.k, self.n, concentration, symmetric=True).marginal
            else:
                newmarginal = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)

        # replace the cause or the effect by this marginal
        if on == 'cause':
            return CategoricalStatic(newmarginal, self.conditional)
        else:  # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            return CategoricalStatic(self.marginal, newconditional)

    def sample(self, m, return_tensor=False):
        """For each of the n distributions, return m samples. (n*m*2 array) """
        flatjoints = self.to_joint().reshape((self.n, self.k ** 2))
        samples = np.array([np.random.choice(self.k ** 2, size=m, p=p) for p in flatjoints])
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
    for intervention in ['cause', 'effect', 'independent']:
        for symmetric_init in [True, False]:
            for symmetric_intervention in [True, False]:
                experiment(2, 3, 1, intervention, symmetric_init, symmetric_intervention)


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
        return F.log_softmax(
            (self.sba + self.sa.unsqueeze(dim=2)).view(self.n, self.k ** 2),
            dim=1
        ).view(self.n, self.k, self.k)

    def to_static(self):
        return CategoricalStatic(
            logit2proba(self.sa.detach().numpy()),
            logit2proba(self.sba.detach().numpy())
        )

    def kullback_leibler(self, other):
        joint = self.to_joint()
        return torch.sum((joint - other.to_joint()) * torch.exp(joint), dim=(1, 2))

    def scoredist(self, other):
        return torch.sum((self.sa - other.sa) ** 2, dim=1) \
               + torch.sum((self.sba - other.sba) ** 2, dim=(1, 2))

    def __repr__(self):
        return f"CategoricalModule(joint={self.to_joint().detach()})"


def test_CategoricalModule(n=7, k=5):
    print('test categorical module')
    references = sample_joint(k, n, 1)
    intervened = references.intervention(on='cause', concentration=1)

    modules = CategoricalModule(references.sa, references.sba)
    optimizer = optim.SGD(modules.parameters(), lr=1)

    aa, bb = intervened.sample(13, return_tensor=True)
    negativeloglikelihoods = -modules(aa, bb).mean()
    optimizer.zero_grad()
    negativeloglikelihoods.backward()
    optimizer.step()

    imodules = intervened.to_module()
    imodules.kullback_leibler(modules)
    imodules.scoredist(modules)

    # test that reverse is numerically stable
    kls = references.reverse().reverse().to_module().kullback_leibler(modules)
    assert torch.allclose(torch.zeros(n, dtype=torch.double), kls), kls


def experiment_optimize(k, n, T, lr, concentration, intervention,
                        is_init_symmetric=True, is_intervention_symmetric=False,
                        batch_size=10, scheduler_exponent=0, log_interval=10):
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
    causal = sample_joint(k, n, concentration, is_init_symmetric)
    transfer = causal.intervention(
        on=intervention,
        concentration=concentration,
        fromjoint=is_intervention_symmetric
    )
    anticausal = causal.reverse()
    antitransfer = transfer.reverse()

    causal = causal.to_module()
    anticausal = anticausal.to_module()
    transfer = transfer.to_module()
    antitransfer = antitransfer.to_module()

    causaloptimizer = optim.SGD(causal.parameters(), lr=lr)
    antioptimizer = optim.SGD(anticausal.parameters(), lr=lr)

    print(transfer.kullback_leibler(causal)
          - antitransfer.kullback_leibler(anticausal))
    return
    steps = [0]
    with torch.no_grad():
        kl_causal = [transfer.kullback_leibler(causal)]
        scoredist_causal = [transfer.scoredist(causal)]
        kl_anti = [antitransfer.kullback_leibler(anticausal)]
        scoredist_anti = [antitransfer.scoredist(anticausal)]

    for t in tqdm.tqdm(range(1, T + 1)):
        # aa, bb = transfer.sample(m=batch_size, return_tensor=True)

        causaloptimizer.lr = lr / t ** scheduler_exponent
        causaloptimizer.zero_grad()
        # causalloss = - causalmodules(aa, bb).sum() / batch_size
        causalloss = transfer.kullback_leibler(causal).sum()
        causalloss.backward()
        causaloptimizer.step()

        antioptimizer.lr = lr / t ** scheduler_exponent
        antioptimizer.zero_grad()
        # antiloss = - antimodules(bb, aa).sum() / batch_size
        antiloss = antitransfer.kullback_leibler(anticausal).sum()
        antiloss.backward()
        antioptimizer.step()

        if t % log_interval == 0:
            steps.append(t)

            with torch.no_grad():
                kl_causal.append(transfer.kullback_leibler(causal))
                scoredist_causal.append(transfer.scoredist(causal))

                kl_anti.append(antitransfer.kullback_leibler(anticausal))
                scoredist_anti.append(antitransfer.scoredist(anticausal))

    return {
        'steps': steps,
        'kl_causal': torch.stack(kl_causal).numpy(),
        'scoredist_causal': torch.stack(scoredist_causal).numpy(),
        'kl_anti': torch.stack(kl_anti).numpy(),
        'scoredist_anti': torch.stack(scoredist_anti).numpy()
    }


def test_experiment_optimize():
    experiment_optimize(k=7, n=13, T=100, lr=.1, concentration=1, intervention='cause')


if __name__ == "__main__":
    test_proba2logit()
    test_ConditionalStatic()
    test_experiment()
    test_CategoricalModule()
    test_experiment_optimize()
