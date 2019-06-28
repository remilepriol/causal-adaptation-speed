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
        trajectory = categorical_distance.experiment_optimize(
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


def parameter_sweep(intervention, k=10, seed=17):
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500,
        'batch_size': 1,
        'is_init_symmetric': False,
        'is_intervention_symmetric': False,
        'concentration': 1, 'intervention': intervention
    }
    for exponent in [0, .5]:
        for lr, n0 in zip(np.logspace(-3, 1, 10),
                          np.logspace(0, 3, 10)):
            np.random.seed(seed)
            trajectory = categorical_distance.experiment_optimize(
                lr=lr, n0=n0, **base_experiment)
            experiment = {'lr': lr, 'n0': n0,
                          'scheduler_exponent': exponent,
                          **base_experiment, **trajectory}
            results.append(experiment)

    savedir = 'results'
    os.makedirs(savedir, exist_ok=True)
    savefile = f'parameter_sweep_{intervention}_k={k}.pkl'

    if base_experiment['is_init_symmetric']:
        savefile = 'syminit_' + savefile
    else:
        savefile = 'asyminit_' + savefile
    if base_experiment['is_intervention_symmetric']:
        savefile = 'syminter_' + savefile
    else:
        savefile = 'asyminter_' + savefile

    savepath = os.path.join(savedir, savefile)
    with open(savepath, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    # optimize_distances()
    parameter_sweep('cause')
    parameter_sweep('effect')
