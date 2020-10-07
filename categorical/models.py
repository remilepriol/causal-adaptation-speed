import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn, optim

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
        elif on == 'singlecond':
            newscores = stats.loggamma.rvs(concentration, size=(self.n, self.k))
            newscores -= newscores.mean(1, keepdims=True)
            # if 'simple':
            #     a0 = 0
            # elif 'max':  # TODO
            #     a0 = np.argmax(self.sa, axis=1)
            #     a0 = np.random.choice()

            a0 = np.argmax(self.sa, axis=1)
            sba = self.sba.copy()
            sba[np.arange(self.n), a0] = newscores
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


class CategoricalModule(nn.Module):
    """Represent n categorical conditionals as a pytorch module"""

    def __init__(self, sa, sba, is_btoa=False):
        super(CategoricalModule, self).__init__()
        self.n, self.k = tuple(sa.shape)

        sa = sa.clone().detach() if torch.is_tensor(sa) else torch.tensor(sa)
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
        batch_size = a.shape[1]
        if self.BtoA:
            a, b = b, a
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size)
        return self.to_joint()[rows.view(-1), a.view(-1), b.view(-1)].view(self.n, batch_size)

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
        # if self.k ** 2 != k2:
        #     raise ValueError('Logits matrix can not be reshaped to square.')

        # normalize to sum to 0
        logits = logits - logits.mean(dim=1, keepdim=True)
        self.logits = nn.Parameter(logits)

    @property
    def logpartition(self):
        return torch.logsumexp(self.logits, dim=1)

    def forward(self, a, b):
        batch_size = a.shape[1]
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size).view(-1)
        index = (a * self.k + b).view(-1)
        return F.log_softmax(self.logits, dim=1)[rows, index].view(self.n, batch_size)

    def kullback_leibler(self, other):
        a = self.logpartition
        kl = torch.sum((self.logits - other.logits) * torch.exp(self.logits - a[:, None]), dim=1)
        return kl - a + other.logpartition

    def scoredist(self, other):
        return torch.sum((self.logits - other.logits) ** 2, dim=1)

    def __repr__(self):
        return f"CategoricalJoint(logits={self.logits.detach()})"


class JointMAP:

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
    mle = JointMAP(static.n, static.k)
    mle.counts = n0 * torch.from_numpy(static.to_joint(return_probas=True))
    return mle


if __name__ == "__main__":
    test_ConditionalStatic()
    test_CategoricalModule()
