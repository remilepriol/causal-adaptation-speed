import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm


def bigplot(exp):
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

    confidence = (5, 95)
    for ax, ratio in zip([axs[2, 0], axs[2, 1]], [ratio_proba, ratio_score]):
        ax.plot(dimensions, np.mean(ratio, axis=-1), label='mean')
        ax.fill_between(dimensions, np.percentile(ratio, confidence[0], axis=-1),
                        np.percentile(ratio, confidence[1], axis=-1),
                        alpha=.4, label='confidence {} %'.format(confidence[1] - confidence[0]))

    for ax in axs.flatten():
        ax.legend()

    return fig, axs


def plotname(exp):
    return "{}_{}_{}_{}.pdf".format(
        exp['intervention'],
        exp['concentration'],
        exp['symmetric_init'],
        exp['symmetric_intervention']
    )


if __name__ == "__main__":

    with open('results/categorical_distances.pkl', 'rb') as fin:
        results = pickle.load(fin)

    plotdir = 'plots/categorical'
    os.makedirs(plotdir, exist_ok=True)

    for exp in tqdm.tqdm(results):
        bigplot(exp)
        plt.savefig(os.path.join(plotdir, plotname(exp)))
        plt.close()
