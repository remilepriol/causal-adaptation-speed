from collections import defaultdict
import os
import pickle
from normal_pkg import normal


def intervention_distances(k, n, intervention='cause', init='natural'):
    """Sample  n conditional Gaussians between cause and effect of dimension k
    and evaluate the distance after intervention between causal and anticausal models."""

    ans = defaultdict(list)
    for i in range(n):
        # sample mechanisms
        if init == 'natural':
            original = normal.sample_natural(k)
        elif init == 'cholesky':
            original = normal.sample_cholesky(k).to_natural()

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


def record_distances():
    n = 300
    kk = [10, 20]

    results = []
    for intervention in ['cause', 'effect']:
        for init in ['natural', 'cholesky']:
            for k in kk:
                exp = {
                    'intervention': intervention,
                    'init': init,
                    'dim': k,
                }
                print(exp)
                exp = {**exp, 'distances': intervention_distances(k, n, intervention, init)}
                results.append(exp)

    savedir = 'normal_results'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(
            savedir, f'distances_{n}.pkl'), 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == '__main__':
    a = intervention_distances(3, 4)
    record_distances()
