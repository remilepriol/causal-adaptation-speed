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

    def sample(self, n):
        aa = np.random.multivariate_normal(self.mua, self.cova, size=n)
        bb = np.dot(aa, self.linear.T) \
             + np.random.multivariate_normal(self.bias, self.covcond, size=n)
        return aa, bb


class MeanJoint:

    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov

    def to_natural(self):
        precision = np.linalg.inv(self.cov)
        return NaturalJoint(precision @ self.mu, precision)

    def sample(self, n):
        return np.random.multivariate_normal(self.mu, self.cov, size=n)

    def encode(self, encoder):
        mu = np.dot(encoder, self.mu)
        cov = np.linalg.multi_dot([encoder, self.cov, encoder.T])
        return MeanJoint(mu, cov)


class NaturalJoint:

    def __init__(self, eta, precision):
        self.eta = eta
        self.precision = precision

    def to_mean(self):
        cov = np.linalg.inv(self.precision)
        return MeanJoint(cov @ self.eta, cov)

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
        return NaturalJoint(eta, precision)

    @property
    def logpartition(self):
        s, logdet = np.linalg.slogdet(self.precision)
        assert s == 1
        return self.eta.T @ np.linalg.solve(self.precision, self.eta) - logdet

    def negativeloglikelihood(self, x):
        """Return the NLL of each point in x.
        x is a n*2dim array where each row is a datapoint.
        """
        linearterm = -self.x @ self.eta - np.sum((self.x @ self.precision) * self.x, axis=1)
        return linearterm + self.logpartition


class NaturalConditionalNormal:
    """Joint Gaussian distribution between a cause variable A and an effect variable B.

    B is a linear encoder of A plus gaussian noise.
    The relevant parameters to describe the joint distribution are the parameters of A,
    and the parameters of B given A.
    """

    def __init__(self, etaa, preca, linear, bias, preccond):
        self.dim = etaa.shape[0]
        # marginal
        self.etaa = etaa
        self.preca = preca
        # conditional
        self.linear = linear
        self.bias = bias
        self.preccond = preccond

    def to_mean(self):
        cova = np.linalg.inv(self.preca)
        covcond = np.linalg.inv(self.preccond)
        mua = cova @ self.etaa
        linear = covcond @ self.linear
        bias = covcond @ self.bias
        return MeanConditionalNormal(mua, cova, linear, bias, covcond)

    def to_cholesky(self):
        la = np.linalg.cholesky(self.preca)
        lcond = np.linalg.cholesky(self.preccond)
        lainv = np.linalg.inv(la)
        lcondinv = np.linalg.inv(lcond)
        return CholeskyConditionalNormal(
            za=np.dot(lainv, self.etaa),
            la=la,
            linear=np.dot(lcondinv, self.linear),
            bias=np.dot(lcondinv, self.bias),
            lcond=lcond
        )

    def intervention(self, on):
        """Sample natural parameters of a marginal distribution
        and substitute them in the cause or effect marginals."""
        eta = np.random.randn(self.dim)
        prec = wishart(self.dim).rvs()
        if on == 'cause':
            return NaturalConditionalNormal(eta, prec, self.linear, self.bias, self.preccond)
        elif on == 'effect':
            return NaturalConditionalNormal(
                self.etaa, self.preca, np.zeros_like(self.linear), eta, prec)

    def to_joint(self):
        tmp = np.dot(self.linear.T, np.linalg.inv(self.preccond))
        eta = np.concatenate([self.etaa - np.dot(tmp, self.bias), self.bias], axis=0)

        d = self.dim
        precision = np.zeros([2 * d, 2 * d])
        precision[:d, :d] = self.preca + np.dot(tmp, self.linear)
        precision[:d, d:] = - self.linear.T
        precision[d:, :d] = - self.linear
        precision[d:, d:] = self.preccond
        return NaturalJoint(eta, precision)

    def reverse(self):
        """Return the ConditionalGaussian from B to A."""
        return self.to_joint().reverse().to_conditional()

    def squared_distance(self, other):
        """Return squared euclidean distance in natural parameter space."""
        return (
                np.sum((self.etaa - other.etaa) ** 2)
                + np.sum((self.preca - other.preca) ** 2)
                + np.sum((self.linear - other.linear) ** 2)
                + np.sum((self.bias - other.bias) ** 2)
                + np.sum((self.preccond - other.condprec) ** 2)
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
            linear=np.dot(self.lcond.T, self.linear),
            bias=np.dot(self.lcond.T, self.bias),
            preccond=np.dot(self.lcond, self.lcond.T)
        )

    def squared_distance(self, other):
        return (
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


def test_Normals(dim=2):
    ConditionalGaussian.random(dim, symmetric=False)
    a = ConditionalGaussian.random(dim, symmetric=True)
    b = a.intervene_on_cause()
    a.intervene_on_effect()
    a.reverse()
    a.squared_distance(b)
    a.sample(10)
    transform = scipy.stats.ortho_group.rvs(2 * dim)
    c = a.encode(transform)
    d = b.encode(transform)

    a.is_consistent()
    b.is_consistent()
    c.is_consistent()
    d.is_consistent()

    # print("causal distance after intervention", a.squared_distances(b))
    # print("transformed after intervention", c.squared_distances(d))

    dist = a.reverse().reverse().squared_distance(a)
    assert np.allclose(dist, 0), print("reverse reverse", dist)
    dist = c.reverse().reverse().squared_distance(c)
    assert np.allclose(dist, 0), print("reverse reverse", dist)
    dist = a.reverse(viajoint=True).squared_distance(a.reverse(viajoint=False))
    assert np.allclose(dist, 0), print("reverse vs joint reverse", dist)


test_ConditionalGaussian()


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

        meandist, natdist = original.squared_distance(transfer)
        revmeandist, revnatdist = revorig.squared_distance(revtrans)

        ans[i] = np.array([meandist, revmeandist, revmeandist / meandist,
                           natdist, revnatdist, revnatdist / natdist])

    return ans


gaussian_distances(3, 4)


def transform_distances(k, n, m, intervention='cause', transformation='orthonormal',
                        noiserange=None):
    """ Evaluate distance induced by interventions and orthonormal transformations.

    Sample  n conditional Gaussians between cause and effect of dimension k.
    For each of these reference distribution, sample an intervention.
    Sample m orthonormal encoder of dimension 2k.
    Get the n*(m+2) transformed distribution for reference and transfer distributions
    +2 because we want to see the values of causal and anticausal models.
    Evaluate the distance after intervention in transformed and non-trnasformed spaces.
    """

    if transformation == 'orthonormal':
        transformers = scipy.stats.ortho_group.rvs(dim=2 * k, size=m)
    else:  # tranformation=='small'
        # small orthonormal deviations around the identity
        noise = np.random.randn(2 * k, 2 * k)
        antisymmetric = noise - noise.T
        if noiserange is None:
            noiserange = np.linspace(0.1, 1, m)
        else:
            m = len(noiserange)
        transformers = [scipy.linalg.expm(eps * antisymmetric) for eps in noiserange]

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

        distances = [d.squared_distance(dt)
                     for d, dt in zip(alloriginals, alltransfers)]

        ans[i] = np.array(distances)

    return ans


transform_distances(3, 4, 5)
transform_distances(3, 4, 5, transformation='small')
transform_distances(3, 4, 5, intervention='effect', transformation='small')
