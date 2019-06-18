import os
import pickle

import numpy as np

import categorical_distance


def all_distances():
    n = 300
    kk = np.arange(2, 100, 8)

    results = []
    for intervention in ['cause', 'effect']:
        for symmetric_init in [True, False]:
            for symmetric_intervention in [True, False]:
                for concentration in [0.1, 1, 10]:
                    distances = []
                    for k in kk:
                        distances.append(categorical_distance.experiment(
                            k, n, 1, intervention, symmetric_init, symmetric_intervention))

                    exp = {
                        'intervention': intervention,
                        'symmetric_init': symmetric_init,
                        'symmetric_intervention': symmetric_intervention,
                        'concentration': concentration,
                        'dimensions': kk,
                        'distances': np.array(distances)
                    }
                    results.append(exp)

    savedir = 'results'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, f'categorical_distances_{n}.pkl'), 'wb') as fout:
        pickle.dump(results, fout)


def optimize_distances(k =10):
    results = []
    base_experiment = {'n': 5, 'k': k, 'T': 3000, 'concentration': 1, 'intervention': 'cause'}
    for lr in [0.05, 0.1, .5]:
        trajectory = categorical_distance.experiment_optimize(lr=lr, **base_experiment)
        experiment = {**base_experiment, 'lr': lr, **trajectory}
        results.append(experiment)

    savedir = 'results'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, f'categorical_optimize_k={k}.pkl'), 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    optimize_distances()
    pass
