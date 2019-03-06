import numpy as np
import scipy
import scipy.linalg
import scipy.stats


class ConditionalGaussian():
    """Joint Gaussian distribution between a cause variable A and an effect variable B.

    B is a linear transformation of A plus gaussian noise.
    The relevant parameters to describe the joint distribution are the parameters of A,
    and the parameters of B given A.
    For both distributions we store both meand natural parameters because we wish to compute
    distances in both of these parametrizations and save compute time.
    """
    addfreedom = 10

    def __init__(self, dim, mua, cova, linear, bias, condcov,
                 etaa=None, preca=None, natlinear=None, natbias=None, condprec=None):
        self.dim = dim
        self.eye = np.eye(dim)

        # mean parameters
        self.mua = mua
        self.cova = cova

        self.linear = linear
        self.bias = bias
        self.condcov = condcov

        # natural parameters
        self.preca = np.linalg.inv(self.cova) if preca is None else preca
        self.etaa = np.dot(self.preca, self.mua) if etaa is None else etaa

        self.condprec = np.linalg.inv(self.condcov) if condprec is None else condprec
        self.natlinear = np.dot(self.condprec, self.linear) if natlinear is None else natlinear
        self.natbias = np.dot(self.condprec, self.bias) if natbias is None else natbias

    def joint_parameters(self):
        mub = np.dot(self.linear, self.mua) + self.bias
        mean = np.concatenate([self.mua, mub], axis=0)

        crosscov = np.dot(self.cova, self.linear.T)
        covariance = np.zeros([2 * self.dim, 2 * self.dim])
        covariance[:self.dim, :self.dim] = self.cova
        covariance[:self.dim, self.dim:] = crosscov
        covariance[self.dim:, :self.dim] = crosscov.T
        covariance[self.dim:, self.dim:] = self.condcov + np.linalg.multi_dot([
            self.linear, self.cova, self.linear])

        return mean, covariance

    @classmethod
    def from_joint(cls, mean, covariance):
        dim = int(mean.shape[0] / 2)

        # parameters of marginal on A
        mua = mean[:dim]
        cova = covariance[:dim, :dim]
        preca = np.linalg.inv(cova)

        # intermediate values required for calculus
        mub = mean[dim:]
        covb = covariance[dim:, dim:]
        crosscov = covariance[:dim, dim:]

        # parameters of conditional
        linear = np.dot(crosscov.T, preca)
        bias = mub - np.dot(linear, mua)
        covcond = covb - np.linalg.multi_dot([linear, cova, linear.T])

        return cls(dim, mua, cova, linear, bias, covcond, preca=preca)

    @classmethod
    def random(cls, dim, symmetric=False):
        """Return a random ConditionalGaussian where each variable has dimension dim.

        If  symmetric is False, then sample each parameters independently.
        Else ensure that cause and effect distributions are sampled similarly
        by sampling the joint and then computing the conditional parameters.

        """
        if not symmetric:
            covariance_distribution = scipy.stats.invwishart(
                df=dim + cls.addfreedom, scale=np.eye(dim))

            # parameters of marginal on A
            mua = np.random.randn(dim)
            cova = covariance_distribution.rvs()

            # parameters of conditional
            linear = np.random.randn(dim, dim)
            bias = np.random.randn(dim)
            covcond = covariance_distribution.rvs()

            return cls(dim, mua, cova, linear, bias, covcond)

        else:
            mean = np.random.randn(2 * dim)
            covariance = scipy.stats.invwishart(
                df=2 * (dim + cls.addfreedom), scale=np.eye(2 * dim)).rvs()

            return cls.from_joint(mean, covariance)

    def intervene_on_cause(self):
        """Random intervention on the mean parameters of the cause A"""
        mua = np.random.randn(self.dim)
        cova = scipy.stats.invwishart(
            df=self.dim + ConditionalGaussian.addfreedom, scale=self.eye).rvs()

        return ConditionalGaussian(
            self.dim, mua, cova, self.linear, self.bias, self.condcov,
            natlinear=self.natlinear, natbias=self.natbias, condprec=self.condprec
        )

    def intervene_on_effect(self):
        """Random intervention on the mean parameters of the effect B"""
        mub = np.random.randn(self.dim)
        covb = scipy.stats.invwishart(
            df=self.dim + ConditionalGaussian.addfreedom, scale=self.eye).rvs()

        return ConditionalGaussian(
            self.dim, self.mua, self.cova, etaa=self.etaa, preca=self.preca,
            linear=np.zeros_like(self.linear), natlinear=np.zeros_like(self.natlinear),
            bias=mub, condcov=covb
        )

    def reverse(self):
        """Return the ConditionalGaussian from B to A."""
        mub = np.dot(self.linear, self.mua) + self.bias
        covb = self.condcov + np.dot(self.linear, np.dot(self.cova, self.linear))

        natlinear = self.natlinear.T
        natbias = self.etaa - np.dot(natlinear, self.bias)
        preccond = self.preca + np.linalg.multi_dot([self.linear.T, self.condprec, self.linear])

        covcond = np.linalg.inv(preccond)
        linear = np.dot(covcond, natlinear)
        bias = np.dot(covcond, natbias)

        return ConditionalGaussian(self.dim, mua=mub, cova=covb,
                                   linear=linear, bias=bias, condcov=covcond,
                                   natlinear=natlinear, natbias=natbias, condprec=preccond)

    def squared_distances(self, other):
        """Return squared distance both in mean and natural parameter space."""
        meandist = (
                np.sum((self.mua - other.mua) ** 2)
                + np.sum((self.cova - other.cova) ** 2)
                + np.sum((self.linear - other.linear) ** 2)
                + np.sum((self.bias - other.bias) ** 2)
                + np.sum((self.condcov - other.condcov) ** 2)
        )

        natdist = (
                np.sum((self.etaa - other.etaa) ** 2)
                + np.sum((self.preca - other.preca) ** 2)
                + np.sum((self.natlinear - other.natlinear) ** 2)
                + np.sum((self.natbias - other.natbias) ** 2)
                + np.sum((self.condprec - other.condprec) ** 2)
        )

        return meandist, natdist

    def sample(self, n):
        aa = np.random.multivariate_normal(self.mua, self.cova, size=n)
        bb = np.dot(aa, self.linear.T) \
             + np.random.multivariate_normal(self.bias, self.condcov, size=n)
        return aa, bb

    def encode(self, matrix):
        mean, covariance = self.joint_parameters()
        newmean = np.dot(matrix, mean)
        newcovariance = np.linalg.multi_dot([
            matrix, covariance, matrix.T
        ])

        return ConditionalGaussian.from_joint(newmean, newcovariance)


dim = 2
a = ConditionalGaussian.random(dim, symmetric=False)
a = ConditionalGaussian.random(dim, symmetric=True)
a.intervene_on_cause()
a.intervene_on_effect()
b = a.reverse()
a.squared_distances(b)
a.sample(10)
transform = scipy.stats.ortho_group.rvs(2 * dim)
c = a.encode(transform)


def gaussian_distances(k, n, intervention='cause', symmetric=False):
    """Sample  n conditional Gaussians between cause and effect of dimension k
    and evaluate the distance after intervention between causal and anticausal models."""

    ans = np.zeros([n, 6])
    for i in range(n):
        # sample mechanisms
        original = ConditionalGaussian.random(k, symmetric)
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


def transform_distances(k, n, m, intervention='cause', transformation='orthonormal'):
    """ Evaluate distance induced by interventions and orthonormal transformations.

    Sample  n conditional Gaussians between cause and effect of dimension k.
    For each of these reference distribution, sample an intervention.
    Sample m orthonormal transformation of dimension 2k.
    Get the n*(m+2) transformed distribution for reference and transfer distributions
    +2 because we want to see the values of causal and anticausal models.
    Evaluate the distance after intervention in transformed and non-trnasformed spaces.
    """

    if transformation == 'orthonormal':
        transformers = scipy.stats.ortho_group.rvs(dim=2 * k, size=m)
    else:  # tranformation=='small'
        # small deviations around the identity
        noise = np.random.randn(2 * k, 2 * k)
        antisymmetric = noise - noise.T
        transformers = [scipy.linalg.expm(eps * antisymmetric) for eps in
                        np.linspace(0.001, 1, m)]

    ans = np.zeros([n, m + 2, 2])
    for i in range(n):
        # sample mechanisms
        original = ConditionalGaussian.random(k, symmetric=True)
        alloriginals = [original, original.reverse()] + [original.encode(t) for t in transformers]

        if intervention == 'cause':
            transfer = original.intervene_on_cause()
        else:  # intervention on effect
            transfer = original.intervene_on_effect()
        alltransfers = [transfer, transfer.reverse()] + [transfer.encode(t) for t in transformers]

        distances = [d.squared_distances(dt)
                     for d, dt in zip(alloriginals, alltransfers)]

        ans[i] = np.array(distances)

    return ans


transform_distances(3, 4, 5)
transform_distances(3, 4, 5, transformation='small')
transform_distances(3, 4, 5, intervention='effect', transformation='small')
