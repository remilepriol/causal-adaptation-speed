import os
import pickle
from collections import defaultdict

import numpy as np

from normal_pkg import normal


def intervention_distances(k, n, intervention='cause', init='natural', interpolation=1):
    """Sample  n conditional Gaussians between cause and effect of dimension k
    and evaluate the distance after intervention between causal and anticausal models."""

    ans = defaultdict(list)
    for i in range(n):
        # sample mechanisms
        reference = normal.sample(k, init)

        try:
            interp = interpolation[i]
        except TypeError:
            interp = interpolation

        transfer = reference.intervention(intervention, interp)

        ans['causal_nat'] += [reference.distance(transfer)]
        revref = reference.reverse()
        revtrans = transfer.reverse()
        ans['anti_nat'] += [revref.distance(revtrans)]
        ans['joint_nat'] += [reference.to_joint().distance(transfer.to_joint())]
        ans['causal_cho'] += [reference.to_cholesky().distance(transfer.to_cholesky())]
        ans['anti_cho'] += [revref.to_cholesky().distance(revtrans.to_cholesky())]

    return ans


def record_distances(savedir = 'normal_results'):
    n = 100
    # kk = [1, 2, 3, 10]
    # kk = [20, 30, 40]
    kk = [10]

    results = []
    for intervention in ['cause', 'effect', 'mechanism']:
        for init in ['natural']:  # , 'cholesky']:
            for k in kk:
                np.random.seed(1)
                exp = {
                    'intervention': intervention,
                    'init': init,
                    'dim': k,
                }
                print('Recording distances for ', exp)
                exp = {**exp, 'distances': intervention_distances(k, n, intervention, init)}
                results.append(exp)

    os.makedirs(savedir, exist_ok=True)
    savefile = os.path.join(savedir, f'distances_{n}.pkl')
    print("Saving results in ", savefile)
    with open(savefile, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == '__main__':
    a = intervention_distances(3, 4)
    record_distances()
