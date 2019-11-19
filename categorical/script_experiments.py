import os
import pickle

import numpy as np

from categorical import distances1


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
                        distances.append(distances1.experiment(
                            k, n, 1, intervention, symmetric_init,
                            symmetric_intervention))

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

    with open(os.path.join(
            savedir, f'categorical_distances_{n}.pkl'), 'wb') as fout:
        pickle.dump(results, fout)


def optimize_distances(k=10):
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500, 'n0': 10,
        'batch_size': 1, 'scheduler_exponent': 0,
        'concentration': 1, 'intervention': 'cause'
    }
    # for lr in [0.01, 0.05, 0.1]:
    for lr in [0.01, 0.1, .5]:
        trajectory = distances1.experiment_optimize(
            lr=lr, **base_experiment)
        experiment = {**base_experiment, 'lr': lr, **trajectory}
        results.append(experiment)

    savedir = 'results'
    os.makedirs(savedir, exist_ok=True)
    savefile = os.path.join(savedir, f'categorical_optimize_k={k}.pkl')
    if os.path.exists(savefile):
        with open(savefile, 'rb') as fin:
            previous_results = pickle.load(fin)
    else:
        previous_results = []

    with open(savefile, 'wb') as fout:
        pickle.dump(previous_results + results, fout)


def parameter_sweep(intervention, k, init, seed=17):
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500,
        'batch_size': 1,
        'intervention': intervention,
        'is_init_symmetric': init,
        'concentration': 1, 'use_map': False
    }
    for exponent in [0]:
        # for lr in [1, 2, 3, 4]:
        for lr in [.03, .1, .3, 1, 3, 9, 30]:
            np.random.seed(seed)
            parameters = {'lr': lr, 'n0': 1, 'scheduler_exponent': exponent, **base_experiment}
            trajectory = distances1.experiment_optimize(**parameters)
            results.append({**parameters, **trajectory})

    savedir = 'results'
    os.makedirs(savedir, exist_ok=True)

    savefile = f'{intervention}_k={k}.pkl'
    if base_experiment['is_init_symmetric']:
        savefile = 'syminit_' + savefile
    else:
        savefile = 'asyminit_' + savefile
    savefile = 'sweep2_' + savefile

    savepath = os.path.join(savedir, savefile)
    with open(savepath, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    # optimize_distances()
    for init in [True, False]:
        for k in [10, 20, 50]:
            parameter_sweep('cause', k, init)
            parameter_sweep('effect', k, init)
            parameter_sweep('geometric', k, init)
            # parameter_sweep('weightedgeo', k, init)
            parameter_sweep('independent', k, init)

