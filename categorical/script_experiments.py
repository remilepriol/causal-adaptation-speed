import os
import pickle

import numpy as np

from categorical.experiment_loops import experiment_guess, experiment_optimize


def optimize_distances(k=10):
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500, 'n0': 10,
        'batch_size': 1, 'scheduler_exponent': 0,
        'concentration': 1, 'intervention': 'cause'
    }
    # for lr in [0.01, 0.05, 0.1]:
    for lr in [0.01, 0.1, .5]:
        trajectory = experiment_optimize(
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


def parameter_sweep(intervention, k, init, seed=17, guess=False, savedir='categorical_results'):
    print(f'intervention on {intervention} with k={k}')
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500,
        'batch_size': 1,
        'intervention': intervention,
        'is_init_dense': init,
        'concentration': 1,
        'use_map': True
    }
    for exponent in [0]:
        for lr, n0 in zip([.03, .1, .3, 1, 3, 9, 30],
                          [0.3, 1, 3, 10, 30, 90, 200]):
            np.random.seed(seed)
            parameters = {'n0': n0, 'lr': lr, 'scheduler_exponent': exponent, **base_experiment}
            if guess:
                trajectory = experiment_guess(**parameters)
            else:
                trajectory = experiment_optimize(**parameters)
            results.append({
                'hyperparameters': parameters,
                'trajectory': trajectory,
                'guess': guess
            })

    os.makedirs(savedir, exist_ok=True)

    savefile = f'{intervention}_k={k}.pkl'
    if base_experiment['is_init_dense']:
        savefile = 'denseinit_' + savefile
    else:
        savefile = 'sparseinit_' + savefile
    if guess:
        savefile = 'guess_' + savefile
    else:
        savefile = 'sweep2_' + savefile

    savepath = os.path.join(savedir, savefile)
    with open(savepath, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    guess = False
    for init_dense in [True, False]:
        for k in [20]:
            # parameter_sweep('cause', k, init_dense, guess=guess)
            # parameter_sweep('effect', k, init_dense, guess=guess)
            # parameter_sweep('singlecond', k, init_dense)
            parameter_sweep('gmechanism', k, init_dense)
