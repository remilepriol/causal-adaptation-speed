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

    def __init__(self, marginal, conditional):
        """The distribution is represented by a marginal p(a) and a conditional p(b|a)

        marginal is n*k array.
        conditional is n*k*k array. Each element conditional[i,j,k] is p_i(b=k |a=j)
        """
        self.n = marginal.shape[0]
        self.k = marginal.shape[1]

        s = conditional.shape
        assert s[0] == self.n
        assert s[1] == self.k and s[2] == self.k
        assert np.allclose(marginal.sum(axis=-1), np.ones(self.n))
        assert np.allclose(conditional.sum(axis=-1), np.ones((self.n, self.k)))

        self.marginal = marginal
        self.conditional = conditional

        self.sa = proba2logit(self.marginal)
        self.sba = proba2logit(self.conditional)

    def to_joint(self):
        return self.conditional * np.expand_dims(self.marginal, axis=-1)

    def reverse(self):
        """Return conditional from b to a.
        Compute marginal pb and conditional pab such that pab*pb = pba*pa.
        """
        joint = self.to_joint()
        joint = np.swapaxes(joint, -1, -2)  # invert variables
        return joint2conditional(joint)

    def probadist(self, other):
        pd = np.sum((self.marginal - other.marginal) ** 2, axis=(-1))
        pd += np.sum((self.conditional - other.conditional) ** 2, axis=(-1, -2))
        return pd

    def scoredist(self, other):
        sd = np.sum((self.sa - other.sa) ** 2, axis=(-1))
        sd += np.sum((self.sba - other.sba) ** 2, axis=(-1, -2))
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

    def sample(self, m):
        """For each of the n distributions, return m samples. (n*m*2 array) """
        flatjoints = self.to_joint().reshape((self.n, self.k ** 2))
        samples = np.array([np.random.choice(self.k ** 2, size=m, p=p) for p in flatjoints])
        a = samples // self.k
        b = samples % self.k
        return a, b

    def __getitem__(self, item):
        return CategoricalStatic(self.marginal[[item]], self.conditional[[item]])

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.n:
            ans = CategoricalStatic(self.marginal[self.i:self.i + 1], self.conditional[self.i:self.i + 1])
            self.i += 1
            return ans
        raise StopIteration()

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
    """Represent 1 categorical conditionala as a pytorch module"""

    def __init__(self, joint: CategoricalStatic):
        super(CategoricalModule, self).__init__()
        self.k = joint.k
        self.sa = nn.Parameter(torch.tensor(joint.sa[0]))
        self.sba = nn.Parameter(torch.tensor(joint.sba[0]))

    def forward(self, a, b):
        return self.to_joint()[a, b]

    def to_joint(self):
        return F.log_softmax((self.sba + self.sa.unsqueeze(dim=1)).view(-1), dim=0).view(self.k, self.k)

    def to_static(self):
        return CategoricalStatic(logit2proba(self.sa.detach().unsqueeze(dim=0).numpy()),
                                 logit2proba(self.sba.detach().unsqueeze(dim=0).numpy()))

    def __repr__(self):
        return f"CategoricalModule(joint={self.to_joint().detach()})"


def test_CategoricalModule():
    print('test categorical module')
    references = sample_joint(5, 10, 1)
    intervened = references.intervention(on='cause', concentration=1)

    modules = [CategoricalModule(r) for r in references]
    allparameters = [param for m in modules for param in m.parameters()]
    optimizer = optim.SGD(allparameters, lr=1)

    aa, bb = intervened.sample(13)
    negativeloglikelihoods = [-m(a, b).mean() for m, a, b in zip(modules, aa, bb)]
    for ll in negativeloglikelihoods:
        optimizer.zero_grad()
        ll.backward()
        optimizer.step()

    updated = [m.to_static() for m in modules]
    kls = [i.kullback_leibler(u) for i, u in zip(intervened, updated)]


def experiment_optimize(k, n, T, lr, concentration, intervention,
                        symmetric_init=True, symmetric_intervention=False):
    """Measure optimization speed and parameters distance.

    Hypothesis: initial distance to optimum is correlated to optimization speed with SGD.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Use SGD to update a causal
    and an anticausal model for T steps. At each step, measure KL
    and distance in scores for causal and anticausal directions.

    :param intervention: takes value 'cause' or 'effect'
    :param symmetric_init: sample the causal parameters such that the marginals on a and b are
    drawn from the same distribution
    :param symmetric_intervention: sample the intervention parameters from the same law as the
    initial parameters
    """
    causal = sample_joint(k, n, concentration, symmetric_init)
    transfer = causal.intervention(
        on=intervention,
        concentration=concentration,
        fromjoint=symmetric_intervention
    )
    anticausal = causal.reverse()
    antitransfer = transfer.reverse()

    causalmodules = [CategoricalModule(d) for d in causal]
    antimodules = [CategoricalModule(d) for d in anticausal]
    modules = causalmodules + antimodules
    allparameters = [param for m in modules for param in m.parameters()]
    optimizer = optim.SGD(allparameters, lr=lr)

    ans = np.zeros([n, T + 1, 2, 2])
    ans[:, 0, 0, 0] = transfer.kullback_leibler(causal)
    ans[:, 0, 0, 1] = causal.scoredist(transfer)

    ans[:, 0, 1, 0] = antitransfer.kullback_leibler(anticausal)
    ans[:, 0, 1, 1] = anticausal.scoredist(antitransfer)

    for t in tqdm.tqdm(range(1, T + 1)):
        aa, bb = transfer.sample(m=10)
        optimizer.lr = lr/t**(2/3)
        optimizer.zero_grad()
        loss = torch.sum(torch.stack([-m(a, b).mean() for m, a, b in zip(causalmodules, aa, bb)]))
        loss = loss + torch.sum(torch.stack([-m(b, a).mean() for m, a, b in zip(antimodules, aa, bb)]))
        loss.backward()
        optimizer.step()

        for i, (trans, module) in enumerate(zip(transfer, causalmodules)):
            static = module.to_static()
            ans[i, t, 0, 0] = trans.kullback_leibler(static)
            ans[i, t, 0, 1] = trans.scoredist(static)

        for i, (trans, module) in enumerate(zip(antitransfer, antimodules)):
            static = module.to_static()
            ans[i, t, 1, 0] = trans.kullback_leibler(static)
            ans[i, t, 1, 1] = trans.scoredist(static)

        if t % 100 == 0:
            current = logit2proba(causalmodules[0].to_joint().detach().numpy().flatten())
            goal = transfer[0].to_joint().flatten()
            # print(current, goal)
            # print(aa * k + bb)
            # print(causalmodules[0](aa[0], bb[0]))

    return ans


def test_experiment_optimize():
    experiment_optimize(k=2, n=1, T=1000, lr=1, concentration=1, intervention='cause')


if __name__ == "__main__":
    test_proba2logit()
    test_ConditionalStatic()
    test_experiment()
    test_CategoricalModule()
    test_experiment_optimize()
