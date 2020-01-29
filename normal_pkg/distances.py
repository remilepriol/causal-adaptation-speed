from collections import defaultdict

from normal_pkg import normal


def intervention_distances(k, n, intervention='cause'):
    """Sample  n conditional Gaussians between cause and effect of dimension k
    and evaluate the distance after intervention between causal and anticausal models."""

    ans = defaultdict(list)
    for i in range(n):
        # sample mechanisms
        original = normal.sample_natural(k)
        transfer = original.intervention(intervention)
        ans['causal_nat'] += [original.distance(transfer)]

        revorig = original.reverse()
        revtrans = transfer.reverse()
        ans['anti_nat'] += [revorig.distance(revtrans)]

        ans['joint_nat'] += [original.to_joint().distance(transfer.to_joint()
                                                          )]
        ans['causal_cho'] += [original.to_cholesky().distance(transfer.to_cholesky())]
        ans['anti_cho'] += [revorig.to_cholesky().distance(revtrans.to_cholesky())]

    return ans


if __name__=='__main__':
    a = intervention_distances(3, 4)
    print(a)
