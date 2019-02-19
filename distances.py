import numpy as np
import scipy
import scipy.stats


def causal2anti(pa, cond_pba):
    """Return marginal pb and conditional pab such that pab*pb = pba*pa.

    pa is n*k array.
    pba is n*k*k array.
    """
    joint = cond_pba * np.expand_dims(pa, axis=-1)
    joint = np.swapaxes(joint, -1, -2)
    pb = np.sum(joint, axis=-1)
    cond_pab = joint / np.expand_dims(pb, axis=-1)

    return pb, cond_pab


def test_causal2anti():
    pa = np.array([.5, .5])
    pba = np.array([[.5, .5], [0, 1]])

    anspb = np.array([.25, .75])
    anspab = np.array([[1, 0], [1 / 3, 2 / 3]])
    pb, pab = causal2anti(pa, pba)

    assert np.allclose(pb, anspb)
    assert np.allclose(pab, anspab)


test_causal2anti()


def proba2logit(p):
    s = np.log(p)
    s -= np.mean(s, axis=-1, keepdims=True)
    return s


def normal2anti(p):
    pass


def categorical_distances_cause(k, n, concentration=1):
    """Sample n mechanisms from a dirichlet of order k then evaluate the distance after
    intervention on the cause between causal and anticausal models.
    """

    # sample mechanisms
    pa = np.random.dirichlet(concentration * np.ones(k), size=n)
    tpa = np.random.dirichlet(concentration * np.ones(k), size=n)  # transfer / intervention
    pba = np.random.dirichlet(concentration * np.ones(k), size=[n, k])

    # evaluate anticausal probabilities
    pb, pab = causal2anti(pa, pba)
    tpb, tpab = causal2anti(tpa, pba)

    # compute distances
    distances = np.zeros([n, 8])

    distances[:, 0] = np.sum((pa - tpa) ** 2, axis=-1)
    bmarginaldistance = np.sum((pb - tpb) ** 2, axis=-1)
    distances[:, 1] = bmarginaldistance + np.sum((pab - tpab) ** 2, axis=(-1, -2))
    distances[:, 2] = distances[:, 1] / distances[:, 0]

    distances[:, 6] = bmarginaldistance

    # in the score / logits space, we have to consider logits which sum to 0
    sa = proba2logit(pa)
    tsa = proba2logit(tpa)
    # sba = proba2logit(pba)

    sb = proba2logit(pb)
    sab = proba2logit(pab)

    tsb = proba2logit(tpb)
    tsab = proba2logit(tpab)

    # distances
    distances[:, 3] = np.sum((sa - tsa) ** 2, axis=-1)  # causal
    bmarginaldistance = np.sum((sb - tsb) ** 2, axis=-1)
    distances[:, 4] = bmarginaldistance + np.sum((sab - tsab) ** 2, axis=(-1, -2))  # anticausal
    distances[:, 5] = distances[:, 4] / distances[:, 3]  # ratio

    distances[:, 7] = bmarginaldistance

    return distances


categorical_distances_cause(3, 4)


def categorical_distances_effect(k, n, concentration=1):
    """Sample n mechanisms from a dirichlet of order k then evaluate the distance after
    intervention on the effect between causal and anticausal models.
    """

    # sample causal mechanisms
    pa = np.random.dirichlet(concentration * np.ones(k), size=n)
    pba = np.random.dirichlet(concentration * np.ones(k), size=[n, k])
    # evaluate anticausal probabilities
    pb, pab = causal2anti(pa, pba)

    # intervention on the effect. tp = transfer distribution. tp(a,b) = tp(a)*tp(b)
    tpb = np.random.dirichlet(concentration * np.ones(k), size=n)
    tpa = pa
    tpba = np.expand_dims(tpb, axis=1)
    tpab = np.expand_dims(tpa, axis=1)

    # compute distances
    distances = np.zeros([n, 8])

    distances[:, 0] = np.sum((pba - tpba) ** 2, axis=(1, 2))  # causal
    bmarginaldistance = np.sum((pb - tpb) ** 2, axis=1)
    distances[:, 1] = bmarginaldistance + np.sum((pab - tpab) ** 2, axis=(1, 2))
    distances[:, 2] = distances[:, 1] / distances[:, 0]

    distances[:, 6] = bmarginaldistance

    # in the score / logits space, we have to consider logits which sum to 0
    sba = proba2logit(pba)
    tsba = proba2logit(tpba)

    sb = proba2logit(pb)
    tsb = proba2logit(tpb)

    sab = proba2logit(pab)
    tsab = proba2logit(tpab)

    # distances
    distances[:, 3] = np.sum((sba - tsba) ** 2, axis=(1, 2))
    bmarginaldistance = np.sum((sb - tsb) ** 2, axis=1)
    distances[:, 4] = bmarginaldistance + np.sum((sab - tsab) ** 2, axis=(1, 2))
    distances[:, 5] = distances[:, 4] / distances[:, 3]

    distances[:, 7] = bmarginaldistance

    return distances


categorical_distances_effect(3, 4)


class ConditionalGaussian():
    """Joint Gaussian distribution between a cause variable A and an effect variable B.

    B is a linear transformation of A plus gaussian noise.
    The relevant parameters to describe the joint distribution are the parameters of A,
    and the parameters of B given A.
    For both distributions we store both meand natural parameters because we wish to compute
    distances in both of these parametrizations and save compute time.
    """
    addfreedom = 10

    def __init__(self, dim, mua=None, cova=None, linear=None, bias=None, covcond=None,
                 etaa=None, preca=None, natlinear=None, natbias=None, preccond=None):
        self.dim = dim
        self.eye = np.eye(dim)

        # mean parameters
        self.mua = np.zeros(dim) if mua is None else mua
        self.cova = self.eye if cova is None else cova

        self.linear = self.eye if linear is None else linear
        self.bias = np.zeros(dim) if bias is None else bias
        self.covcond = self.eye if covcond is None else covcond

        # natural parameters
        self.preca = np.linalg.inv(self.cova) if preca is None else preca
        self.etaa = np.dot(self.preca, self.mua) if etaa is None else etaa

        self.preccond = np.linalg.inv(self.covcond) if preccond is None else preccond
        self.natlinear = np.dot(self.preccond, self.linear) if natlinear is None else natlinear
        self.natbias = np.dot(self.preccond, self.bias) if natbias is None else natbias

    @staticmethod
    def random(dim):
        """Return a random ConditionalGaussian."""
        covariance_distribution = scipy.stats.invwishart(
            df=dim + ConditionalGaussian.addfreedom, scale=np.eye(dim))

        mua = np.random.randn(dim)
        cova = covariance_distribution.rvs()

        linear = np.random.randn(dim, dim)
        bias = np.random.randn(dim)
        covcond = covariance_distribution.rvs()

        return ConditionalGaussian(dim, mua, cova, linear, bias, covcond)

    def intervene_on_cause(self):
        """Random intervention on the mean parameters of the cause A"""
        mua = np.random.randn(self.dim)
        cova = scipy.stats.invwishart(
            df=self.dim + ConditionalGaussian.addfreedom, scale=self.eye).rvs()

        return ConditionalGaussian(
            self.dim, mua, cova, self.linear, self.bias, self.covcond,
            natlinear=self.natlinear, natbias=self.natbias, preccond=self.preccond
        )

    def intervene_on_effect(self):
        """Random intervention on the mean parameters of the effect B"""
        mub = np.random.randn(self.dim)
        covb = scipy.stats.invwishart(
            df=self.dim + ConditionalGaussian.addfreedom, scale=self.eye).rvs()

        return ConditionalGaussian(
            self.dim, self.mua, self.cova, etaa=self.etaa, preca=self.preca,
            linear=np.zeros_like(self.linear), natlinear=np.zeros_like(self.natlinear),
            bias=mub, covcond=covb
        )

    def reverse(self):
        """Return the ConditionalGaussian from B to A."""
        mub = np.dot(self.linear, self.mua) + self.bias
        covb = self.covcond + np.dot(self.linear, np.dot(self.cova, self.linear))

        natlinear = self.natlinear.T
        natbias = self.etaa - np.dot(natlinear, self.bias)
        preccond = self.preca + np.dot(self.linear.T, np.dot(self.preccond, self.linear))

        covcond = np.linalg.inv(preccond)
        linear = np.dot(covcond, natlinear)
        bias = np.dot(covcond, natbias)

        return ConditionalGaussian(self.dim, mua=mub, cova=covb,
                                   linear=linear, bias=bias, covcond=covcond,
                                   natlinear=natlinear, natbias=natbias, preccond=preccond)

    def squared_distances(self, other):
        """Return squared distance both in mean and natural parameter space."""
        meandist = (
                np.sum((self.mua - other.mua) ** 2)
                + np.sum((self.cova - other.cova) ** 2)
                + np.sum((self.linear - other.linear) ** 2)
                + np.sum((self.bias - other.bias) ** 2)
                + np.sum((self.covcond - other.covcond) ** 2)
        )

        natdist = (
                np.sum((self.etaa - other.etaa) ** 2)
                + np.sum((self.preca - other.preca) ** 2)
                + np.sum((self.natlinear - other.natlinear) ** 2)
                + np.sum((self.natbias - other.natbias) ** 2)
                + np.sum((self.preccond - other.preccond) ** 2)
        )

        return meandist, natdist


a = ConditionalGaussian.random(2)
a.intervene_on_cause()
a.intervene_on_effect()
b = a.reverse()
a.squared_distances(b)


def gaussian_distances(k, n, intervention='cause'):
    """Sample  n conditional gaussians between cause and effect of dimension k
    and evaluate the distance after intervention between causal and anticausal models."""

    ans = np.zeros([n, 6])
    for i in range(n):
        # sample mechanisms
        original = ConditionalGaussian.random(k)
        revorig = original.reverse()

        if intervention == 'cause':
            transfer = original.intervene_on_cause()
        else:  # intervention on effect
            transfer = original.intervene_on_effect()
        revtrans = transfer.reverse()

        meandist, natdist = original.squared_distances(transfer)
        revmeandist, revnatdist = revorig.squared_distances(revtrans)

        ans[i] = np.array([meandist, revmeandist, revmeandist / meandist,
                           natdist, revnatdist, revnatdist / natdist])

    return ans


gaussian_distances(3, 4)
