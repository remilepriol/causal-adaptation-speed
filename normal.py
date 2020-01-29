import numpy as np
import scipy
import scipy.linalg
import scipy.stats


def check_symmetric(a, tol=1e-8):
    if np.allclose(a, a.T, atol=tol):
        return True
    else:
        print(a - a.T)
        return False


class MeanConditionalNormal:

    def __init__(self, mua, cova, linear, bias, covcond):
        self.mua = mua
        self.cova = cova
        self.linear = linear
        self.bias = bias
        self.covcond = covcond

    def to_natural(self):
        # exactly the same formulas as natural to mean
        # those parametrizations are symmetric
        preca = np.linalg.inv(self.cova)
        preccond = np.linalg.inv(self.covcond)
        etaa = preca @ self.mua
        linear = preccond @ self.linear
        bias = preccond @ self.bias
        return NaturalConditionalNormal(etaa, preca, linear, bias, preccond)

    def to_joint(self):
        mub = np.dot(self.linear, self.mua) + self.bias
        mean = np.concatenate([self.mua, mub], axis=0)

        d = self.mua.shape[0]
        crosscov = np.dot(self.cova, self.linear.T)
        cov = np.zeros([2 * d, 2 * d])
        cov[:d, :d] = self.cova
        cov[:d, d:] = crosscov
        cov[d:, :d] = crosscov.T
        cov[d:, d:] = self.covcond + np.linalg.multi_dot([
            self.linear, self.cova, self.linear.T])

        return MeanJointNormal(mean, cov)

    def sample(self, n):
        aa = np.random.multivariate_normal(self.mua, self.cova, size=n)
        bb = np.dot(aa, self.linear.T) \
             + np.random.multivariate_normal(self.bias, self.covcond, size=n)
        return aa, bb


class MeanJointNormal:

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def to_natural(self):
        precision = np.linalg.inv(self.cov)
        return NaturalJointNormal(precision @ self.mean, precision)

    def to_conditional(self):
        d = self.mean.shape[0] // 2

        # parameters of marginal on A
        mua = self.mean[:d]
        cova = self.cov[:d, :d]
        preca = np.linalg.inv(cova)

        # intermediate values required for calculus
        mub = self.mean[d:]
        covb = self.cov[d:, d:]
        crosscov = self.cov[:d, d:]

        # parameters of conditional
        linear = np.dot(crosscov.T, preca)
        bias = mub - np.dot(linear, mua)
        covcond = covb - np.linalg.multi_dot([linear, cova, linear.T])

        return MeanConditionalNormal(mua, cova, linear, bias, covcond)

    def sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def encode(self, encoder):
        mu = np.dot(encoder, self.mean)
        cov = np.linalg.multi_dot([encoder, self.cov, encoder.T])
        return MeanJointNormal(mu, cov)


class NaturalJointNormal:

    def __init__(self, eta, precision):
        self.eta = eta
        self.precision = precision

    def to_mean(self):
        cov = np.linalg.inv(self.precision)
        return MeanJointNormal(cov @ self.eta, cov)

    def to_conditional(self):
        d = self.eta.shape[0] // 2
        # conditional parameters
        preccond = self.precision[d:, d:]
        linear = - self.precision[d:, :d]
        bias = self.eta[d:]

        # marginal parameters
        tmp = linear.T @ np.linalg.inv(preccond)
        preca = self.precision[:d, :d] - tmp @ linear
        etaa = self.eta[:d] + tmp @ bias
        return NaturalConditionalNormal(etaa, preca, linear, bias, preccond)

    def reverse(self):
        d = self.eta.shape[0] // 2
        eta = np.roll(self.eta, d)
        precision = np.roll(self.precision, shift=[d, d], axis=[0, 1])
        return NaturalJointNormal(eta, precision)

    @property
    def logpartition(self):
        s, logdet = np.linalg.slogdet(self.precision)
        assert s == 1
        return self.eta.T @ np.linalg.solve(self.precision, self.eta) - logdet

    def negativeloglikelihood(self, x):
        """Return the NLL of each point in x.
        x is a n*2dim array where each row is a datapoint.
        """
        linearterm = -x @ self.eta - np.sum((x @ self.precision) * x, axis=1)
        return linearterm + self.logpartition


class NaturalConditionalNormal:
    """Joint Gaussian distribution between a cause variable A and an effect variable B.

    B is a linear encoder of A plus gaussian noise.
    The relevant parameters to describe the joint distribution are the parameters of A,
    and the parameters of B given A.
    """

    def __init__(self, etaa, preca, linear, bias, preccond):
        # marginal
        self.etaa = etaa
        self.preca = preca
        # conditional
        self.linear = linear
        self.bias = bias
        self.preccond = preccond

    def to_joint(self):
        tmp = np.dot(self.linear.T, np.linalg.inv(self.preccond))
        eta = np.concatenate([self.etaa - np.dot(tmp, self.bias), self.bias], axis=0)

        d = self.etaa.shape[0]
        precision = np.zeros([2 * d, 2 * d])
        precision[:d, :d] = self.preca + np.dot(tmp, self.linear)
        precision[:d, d:] = - self.linear.T
        precision[d:, :d] = - self.linear
        precision[d:, d:] = self.preccond
        return NaturalJointNormal(eta, precision)

    def to_mean(self):
        cova = np.linalg.inv(self.preca)
        covcond = np.linalg.inv(self.preccond)
        mua = cova @ self.etaa
        linear = covcond @ self.linear
        bias = covcond @ self.bias
        return MeanConditionalNormal(mua, cova, linear, bias, covcond)

    def to_cholesky(self):
        la = np.linalg.cholesky(self.preca)
        assert np.allclose(la @ la.T, self.preca)
        lcond = np.linalg.cholesky(self.preccond)
        return CholeskyConditionalNormal(
            za=np.linalg.solve(la, self.etaa),
            la=la,
            linear=np.linalg.solve(lcond, self.linear),
            bias=np.linalg.solve(lcond, self.bias),
            lcond=lcond
        )

    def intervention(self, on):
        """Sample natural parameters of a marginal distribution
        and substitute them in the cause or effect marginals."""
        eta = np.random.randn(self.etaa.shape[0])
        prec = wishart(self.etaa.shape[0]).rvs()
        if on == 'cause':
            return NaturalConditionalNormal(eta, prec, self.linear, self.bias, self.preccond)
        elif on == 'effect':
            return NaturalConditionalNormal(
                self.etaa, self.preca, np.zeros_like(self.linear), eta, prec)

    def reverse(self):
        """Return the ConditionalGaussian from B to A."""
        return self.to_joint().reverse().to_conditional()

    def distance(self, other):
        """Return Euclidean distance between self and other in natural parameter space."""
        return np.sqrt(
                np.sum((self.etaa - other.etaa) ** 2)
                + np.sum((self.preca - other.preca) ** 2)
                + np.sum((self.linear - other.linear) ** 2)
                + np.sum((self.bias - other.bias) ** 2)
                + np.sum((self.preccond - other.preccond) ** 2)
        )

    @property
    def logpartition(self):
        return self.to_joint().logpartition


class CholeskyConditionalNormal:

    def __init__(self, za, la, linear, bias, lcond):
        self.za = za
        self.la = la
        self.linear = linear
        self.bias = bias
        self.lcond = lcond

    def to_natural(self):
        return NaturalConditionalNormal(
            etaa=np.dot(self.la, self.za),
            preca=np.dot(self.la, self.la.T),
            linear=np.dot(self.lcond, self.linear),
            bias=np.dot(self.lcond, self.bias),
            preccond=np.dot(self.lcond, self.lcond.T)
        )

    def distance(self, other):
        return np.sqrt(
                np.sum((self.za - other.za) ** 2)
                + np.sum((self.la - other.la) ** 2)
                + np.sum((self.linear - other.linear) ** 2)
                + np.sum((self.bias - other.bias) ** 2)
                + np.sum((self.lcond - other.lcond) ** 2)

        )

#  _____      _
# |  __ \    (_)
# | |__) | __ _  ___  _ __ ___
# |  ___/ '__| |/ _ \| '__/ __|
# | |   | |  | | (_) | |  \__ \
# |_|   |_|  |_|\___/|_|  |___/
def wishart(dim, addfreedom=100):
    return scipy.stats.wishart(df=dim + addfreedom, scale=np.eye(dim))


def sample_natural(dim):
    """Sample natural parameters of a ConditionalGaussian of dimension dim."""
    precision_distribution = wishart(dim)

    # parameters of marginal on A
    etaa = np.random.randn(dim)
    preca = precision_distribution.rvs()

    # parameters of conditional
    linear = np.random.randn(dim, dim)
    bias = np.random.randn(dim)
    preccond = precision_distribution.rvs()

    return NaturalConditionalNormal(etaa, preca, linear, bias, preccond)


def sample_triangular(dim):
    t = np.tril(np.random.randn(dim, dim), -1)
    diag = np.sqrt(np.random.gamma(shape=2, scale=2, size=dim))
    return t + np.diag(diag)


def sample_cholesky(dim):
    """Sample cholesky parameters of a ConditionalGaussian of dimension dim."""
    # parameters of marginal on A
    zetaa = np.random.randn(dim)
    lowera = sample_triangular(dim)

    # parameters of conditional
    linear = np.random.randn(dim, dim)
    bias = np.random.randn(dim)
    lowercond = sample_triangular(dim)

    return CholeskyConditionalNormal(zetaa, lowera, linear, bias, lowercond)
