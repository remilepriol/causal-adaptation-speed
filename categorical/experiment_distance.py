import os
import pickle

import numpy as np

from categorical.models import sample_joint


def intervention_distances(k, n, concentration, intervention, dense_init=True):
    """Sample n mechanisms of order k and for each of them sample an intervention on the desired
    mechanism. Return the distance between the original distribution and the intervened
    distribution in the causal parameter space and in the anticausal parameter space.
    """
    # causal parameters
    causal = sample_joint(k, n, concentration, dense_init)
    transfer = causal.intervention(on=intervention, concentration=concentration)
    cpd, csd = causal.sqdistance(transfer)

    # anticausal parameters
    anticausal = causal.reverse()
    antitransfer = transfer.reverse()
    apd, asd = anticausal.sqdistance(antitransfer)

    return np.array([[cpd, csd], [apd, asd]])


def test_intervention_distances():
    print('test experiment')
    for intervention in ['cause', 'effect', 'mechanism', 'gmechanism', 'independent', 'geometric',
                         'weightedgeo']:
        for dense_init in [True, False]:
            intervention_distances(2, 3, 1, intervention, dense_init)


def all_distances(savedir='categorical_results'):
    n = 300
    kk = np.arange(2, 100, 8)

    results = []
    for intervention in ['cause', 'effect']:
        for dense_init in [True, False]:
            for concentration in [0.1, 1, 10]:
                distances = []
                for k in kk:
                    distances.append(intervention_distances(
                        k, n, 1, intervention, dense_init
                    ))
                exp = {
                    'intervention': intervention,
                    'dense_init': dense_init,
                    'concentration': concentration,
                    'dimensions': kk,
                    'distances': np.array(distances)
                }
                results.append(exp)

    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(
            savedir, f'categorical_distances_{n}.pkl'), 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    test_intervention_distances()
