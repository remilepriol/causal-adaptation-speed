import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def abline(ax, slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--')
    

def all_distances():
    with open('normal_results/distances_300.pkl', 'rb') as fin:
        results = pickle.load(fin)

    plotdir = 'plots/distances/'
    os.makedirs(plotdir, exist_ok=True)

    for exp in results:
        if exp['dim']==20:
            name = "{intervention}_init={init}_k={dim}.pdf".format(**exp)
            print(name)
            scatter_distances(exp)
            plt.savefig(os.path.join(plotdir, name))
            plt.close()


def scatter_distances(exp, alpha=.5):
    """Draw 2 scatter plots."""
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax0, ax1 = axs

    dist = exp['distances']
    ax0.scatter(dist['causal_nat'], dist['anti_nat'], label='anti', alpha=alpha)
    ax0.scatter(dist['causal_nat'], dist['joint_nat'], label='joint', alpha=alpha)
    ax1.scatter(dist['causal_cho'], dist['anti_cho'], label='anti', alpha=alpha)

    ax0.set_title('natural parameters')
    ax1.set_title('cholesky parameters')
    for ax in axs:
        ax.legend()
        ax.grid()
        ax.set_xlabel('causal distance')
        ax.set_xlabel('anti distance')
        abline(ax, 1, 0)
            
    return fig


if __name__ == "__main__":
    all_distances()