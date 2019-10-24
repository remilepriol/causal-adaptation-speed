import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm


def dist_plot():
    with open('results/categorical_distances.pkl', 'rb') as fin:
        results = pickle.load(fin)

    plotdir = 'plots/categorical'
    os.makedirs(plotdir, exist_ok=True)

    for exp in tqdm.tqdm(results):
        bigplot(exp)
        plt.savefig(os.path.join(plotdir, bigplotname(exp)))
        plt.close()


def bigplot(exp, confidence=(5, 95)):
    """Draw 4 scatter plots and 2 line plots"""
    proba, score = tuple(np.swapaxes(exp['distances'], 0, 2))
    causal_proba, anti_proba = tuple(proba)
    causal_score, anti_score = tuple(score)

    dimensions = exp['dimensions']

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))

    axs[0, 0].set_title('causal score vs proba')
    axs[0, 1].set_title('anticausal score vs proba')
    axs[1, 0].set_title('proba anticausal vs causal')
    axs[1, 1].set_title('score anticausal vs causal')
    axs[2, 0].set_title('proba ratio anticausal / causal ')
    axs[2, 0].set_title('score ratio anticausal / causal ')

    for k, cpd, apd, csd, asd in zip(
            dimensions,
            causal_proba, anti_proba,
            causal_score, anti_score):
        axs[0, 0].scatter(cpd, csd, label=k, alpha=.2)
        axs[0, 1].scatter(apd, asd, label=k, alpha=.2)
        axs[1, 0].scatter(cpd, apd, label=k, alpha=.2)
        axs[1, 1].scatter(csd, asd, label=k, alpha=.2)

    ratio_proba = anti_proba / causal_proba
    ratio_score = anti_score / causal_score

    for ax, ratio in zip([axs[2, 0], axs[2, 1]], [ratio_proba, ratio_score]):
        ax.plot(dimensions, np.mean(ratio, axis=-1), label='mean')
        ax.fill_between(dimensions,
                        np.percentile(ratio, confidence[0], axis=-1),
                        np.percentile(ratio, confidence[1], axis=-1),
                        alpha=.4, label='confidence {} %'.format(
                confidence[1] - confidence[0]))

    for ax in axs.flatten():
        ax.legend()

    return fig, axs


def bigplotname(exp):
    return "{}_{}_{}_{}.pdf".format(
        exp['intervention'],
        exp['concentration'],
        exp['symmetric_init'],
        exp['symmetric_intervention']
    )


COLORS = {
    'causal': 'blue',
    'anti': 'red',
    'joint': 'yellow',
    'causal_average': 'darkblue',
    'anti_average': 'darkred',
    'joint_average': 'gold',
    'MAP_uniform': 'palegreen',
    'MAP_source': 'darkgreen'
}


def optim_plot():
    with open('results/categorical_optimize_k=10.pkl', 'rb') as fin:
        results = pickle.load(fin)

    plotdir = 'plots/categorical_optim'
    os.makedirs(plotdir, exist_ok=True)

    for exp in results:
        print(longplotname(exp))
        longplot(exp, statistics=True, plot_bound=False)
        plt.savefig(os.path.join(plotdir, 'average_' + longplotname(exp)))
        # curve_plot(exp, statistics=False)
        # plt.savefig(os.path.join(plotdir, 'curves_' + longplotname(exp)))
        optim_scatter(exp)
        plt.savefig(os.path.join(plotdir, 'scatter_' + longplotname(exp)))
        plt.close()


def longplot(exp, confidence=(5, 95), statistics=True,
             plot_bound=False):
    """Draw mean trajectory plot with percentiles"""
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    for ax, metric in zip(axs, ['kl', 'scoredist']):
        ax.grid(True)
        ax.set_yscale('log')
        for model_family in COLORS.keys():
            key = metric + '_' + model_family
            if key in exp:
                values = exp[key]

                if statistics:  # plot mean and percentile statistics
                    ax.plot(
                        exp['steps'],
                        values.mean(axis=1),
                        label=model_family,
                        color=COLORS[model_family]
                    )
                    ax.fill_between(
                        exp['steps'],
                        np.percentile(values, confidence[0], axis=1),
                        np.percentile(values, confidence[1], axis=1),
                        alpha=.4,
                        color=COLORS[model_family]
                    )
                else:
                    ax.plot(
                        exp['steps'],
                        values,
                        alpha=.1,
                        color=COLORS[model_family],
                        label=model_family
                    )

    axs[0].set_ylabel('KL(transfer, model)')
    if statistics:
        axs[0].legend()
    axs[1].set_ylabel('||transfer - model ||^2')

    if plot_bound:
        make_bound(exp, axs[0])


def make_bound(exp, ax):
    for model_family in ['causal', 'anti']:
        initial_distance = exp['scoredist_' + model_family][0].mean()
        steps = np.array(exp['steps'])[1:]
        if exp['batch_size'] == 'full' and exp['scheduler_exponent'] == 0:
            constant = rate_constant_fullbatch(
                smoothness=1 / 2,
                initial_lr=exp['lr'],
                initial_distance=initial_distance
            )
            bound = constant / (steps - 1)
        elif np.isclose(exp['scheduler_exponent'], 2 / 3):
            constant = rate_constant_twothird(
                smoothness=1 / 2,
                initial_lr=exp['lr'],
                initial_distance=initial_distance,
                variance=1 / exp['batch_size']
            )
            bound = constant / (steps ** (1 / 3) - 1)
        else:
            print('Convergence bound only available for '
                  'gradient descent with constant learning rate '
                  'or stochastic gradient descent with learning rate scheduling exponent 2/3.')
            return

        initial_kl = exp['kl_' + model_family][0].mean()
        print(f"{model_family} initial distance = {initial_distance:.2f} \t"
              f" bound constant = {constant:.2f} \t"
              f"ratio = {constant / initial_distance:.2f} \t"
              f"ratio constant / initial kl = {constant / initial_kl:.2f}")

        ax.plot(steps, bound,
                color=COLORS[model_family],
                linestyle='--')


def longplotname(exp):
    return (
        '{intervention}_k={k}_bs={batch_size}_rate={scheduler_exponent:.1f}'
        '_lr={lr}_concentration={concentration}_T={T}.pdf').format(**exp)


def rate_constant_twothird(smoothness, initial_lr, initial_distance, variance):
    return (np.exp(3 * (2 * smoothness * initial_lr) ** 2)
            * (1 + 4 * (smoothness * initial_lr) ** (3 / 2))
            / (3 * initial_lr)
            * (initial_distance + variance / smoothness ** 2))


def rate_constant_fullbatch(smoothness, initial_lr, initial_distance):
    return - initial_distance / (
            initial_lr * (smoothness / 2 * initial_lr - 1))


def optim_scatter(exp, end_step=1000):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))

    index = np.searchsorted(exp['steps'], end_step)
    initial_distances = {}
    ends = {}
    integrals = {}
    for model_family in COLORS.keys():
        key = 'scoredist_' + model_family
        if key in exp:
            initial_distances[model_family] = exp[key][0]
            values = exp['kl_' + model_family]
            ends[model_family] = values[index]
            integrals[model_family] = values[:index].mean(axis=0)

            axs[0].scatter(
                initial_distances[model_family],
                ends[model_family],
                alpha=.3,
                color=COLORS[model_family],
                label=model_family
            )
            axs[1].scatter(
                initial_distances[model_family],
                integrals[model_family],
                alpha=.3,
                color=COLORS[model_family],
                label=model_family
            )

    for ax, metrics in zip(axs, [ends, integrals]):
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xscale('log')
        # plot edge between identical problems
        # ax.plot(
        #     np.stack((
        #         initial_distances['causal'],
        #         initial_distances['anti']
        #     ), axis=1).T,
        #     np.stack((
        #         metrics['causal'],
        #         metrics['anti']
        #     ), axis=1).T,
        #     alpha=.1, color='black')

    axs[0].set_ylabel(f'KL(transfer, model) at step {end_step}')
    axs[0].legend()
    axs[1].set_ylabel(f'Average KL(transfer, model) from step 0 to {end_step}')
    axs[1].set_xlabel(
        'Parameter squared distance from initialization to optimum.')


if __name__ == "__main__":
    optim_plot()
