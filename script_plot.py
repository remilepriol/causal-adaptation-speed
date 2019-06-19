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
        ax.fill_between(dimensions, np.percentile(ratio, confidence[0], axis=-1),
                        np.percentile(ratio, confidence[1], axis=-1),
                        alpha=.4, label='confidence {} %'.format(confidence[1] - confidence[0]))

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


def optim_plot():
    with open('results/categorical_optimize_k=10.pkl', 'rb') as fin:
        results = pickle.load(fin)

    plotdir = 'plots/categorical_optim'
    os.makedirs(plotdir, exist_ok=True)

    for exp in tqdm.tqdm(results):
        longplot(exp, statistics=True)
        plt.savefig(os.path.join(plotdir, 'average_'+longplotname(exp)))
        longplot(exp, statistics=False)
        plt.savefig(os.path.join(plotdir, 'curves_' + longplotname(exp)))
        plt.close()


def longplot(exp, confidence=(5, 95), statistics=True):
    """Draw mean trajectory plot with percentiles"""
    colors = {'causal':'blue', 'anti':'red'}
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    for ax, metric in zip(axs, ['kl', 'scoredist']):
        ax.grid(True)
        ax.set_yscale('log')
        for model_family in ['causal', 'anti']:
            values = exp[metric + '_' + model_family]

            if statistics:  # plot mean and percentile statistics
                ax.plot(
                    exp['steps'],
                    values.mean(axis=1),
                    label=model_family
                )
                ax.fill_between(
                    exp['steps'],
                    np.percentile(values, confidence[0], axis=1),
                    np.percentile(values, confidence[1], axis=1),
                    alpha=.4,
                    label='confidence {} %'.format(
                        confidence[1] - confidence[0])
                )
            else:
                ax.plot(
                    exp['steps'],
                    values,
                    alpha=.1,
                    color=colors[model_family],
                    label=model_family
                )

    axs[0].set_ylabel('KL(transfer, model)')
    if statistics:
        axs[0].legend()
    axs[1].set_ylabel('||transfer - model ||^2')


def longplotname(exp):
    return "{intervention}_k={k}_lr={lr}_bs={batch_size}_concentration={concentration}_T={T}.pdf".format(**exp)


if __name__ == "__main__":
    optim_plot()
