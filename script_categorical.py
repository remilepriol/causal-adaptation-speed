import pickle

import numpy as np

import categorical_distance

if __name__ == "__main__":

    n = 100
    kk = np.arange(2, 100, 8)

    results = []
    for intervention in ['cause', 'effect']:
        for symmetric_init in [True, False]:
            for symmetric_intervention in [True, False]:
                for concentration in [0.1, 1, 10]:
                    for k in kk:
                        distances = categorical_distance.experiment(
                            k, n, 1, intervention, symmetric_init, symmetric_intervention)
                        exp = {
                            'intervention': intervention,
                            'symmetric_init': symmetric_init,
                            'symmetric_intervention': symmetric_intervention,
                            'concentration': concentration,
                            'dimension': k,
                            'distances': distances
                        }
                        results.append(exp)

    with open('results/categorical_distances.pkl', 'wb') as fout:
        pickle.dump(results, fout)
