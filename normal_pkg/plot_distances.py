import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def abline(ax, slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--', color='grey')


def all_distances():
    with open('normal_results/distances_100.pkl', 'rb') as fin:
        results = pickle.load(fin)

    plotdir = 'plots/normal_distances/'
    os.makedirs(plotdir, exist_ok=True)

    for exp in results:
        for unit in ['nat', 'cho']:
            name = "dist{unit}_{intervention}_{init}_k={dim}.pdf".format(unit=unit, **exp)
            savefile = os.path.join(plotdir, name)
            print("Saving in ", savefile)
            scatter_distances(unit, exp)
            plt.tight_layout()
            plt.savefig(savefile)
            plt.close()


def scatter_distances(unit, exp, alpha=.5):
    """Draw anticausal vs causal scatter plot"""
    dist = exp['distances']
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(dist['causal_' + unit], dist['anti_' + unit], linewidth=0, alpha=alpha)
    ax.set_xlabel(r'$|| \theta^{(0)}_{\rightarrow} - \theta^*_{\rightarrow} ||$')
    ax.set_ylabel(r'$|| \theta^{(0)}_{\leftarrow} - \theta^*_{\leftarrow} ||$')
    abline(ax, 1, 0)
    ax.grid()
    ax.axis('equal')

    return fig


if __name__ == "__main__":
    all_distances()
