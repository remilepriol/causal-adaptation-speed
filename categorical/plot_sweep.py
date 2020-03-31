import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

np.set_printoptions(precision=2)


def add_capitals(dico):
    return {**dico, **{key[0].capitalize() + key[1:]: item for key, item in dico.items()}}


COLORS = {
    'causal': 'blue',
    'anti': 'red',
    'joint': 'green',
    'causal_average': 'darkblue',
    'anti_average': 'darkred',
    'joint_average': 'darkgreen',
    'MAP_uniform': 'yellow',
    'MAP_source': 'gold',
    # guess
    'CausalGuessX': 'skyblue',
    'CausalGuessY': 'darkcyan',
    'AntiGuessX': 'salmon',
    'AntiGuessY': 'chocolate',
}
MARKERS = {key: 'o' for key in COLORS}
MARKERS['causal'] = '^'
MARKERS['anti'] = 'v'

COLORS = add_capitals(COLORS)
MARKERS = add_capitals(MARKERS)


def value_at_step(trajectory, nsteps=1000):
    """Return the KL and the integral KL up to nsteps."""
    steps = trajectory['steps']
    index = np.searchsorted(steps, nsteps) - 1

    ans = {}
    # ans['end_step'] = steps[index]
    for key, item in trajectory.items():
        if key.startswith('kl_'):
            ans[key[3:]] = item[index].mean()
            # ans['endkl_' + key[3:]] = item[index].mean()
            # ans['intkl_' + key[3:]] = item[:index].mean()

    return ans


def get_best(results, nsteps):
    """Store per model each parameter and kl values
    then for each model return the argmax parameters and curves
    for kl and integral kl
    """

    by_model = {}
    # dictionary where each key is a model,
    # and each value is a list of this model's hyperparameter
    # and outcome at step nsteps
    for exp in results:
        trajectory = exp['trajectory']
        for model, metric in value_at_step(trajectory, nsteps).items():
            if model not in by_model:
                by_model[model] = []
            toadd = {
                'hyperparameters': exp['hyperparameters'],
                **exp['hyperparameters'],
                'value': metric,
                'kl': trajectory['kl_' + model],
                'steps': trajectory['steps']
            }
            if 'scoredist_' + model in trajectory:
                toadd['scoredist'] = trajectory['scoredist_' + model]
            by_model[model] += [toadd]

    # select only the best hyperparameters for this model.
    for model, metrics in by_model.items():
        dalist = sorted(metrics, key=lambda x: x['value'])
        # Ensure that the optimal configuration does not diverge as optimization goes on.
        for duh in dalist:
            if duh['kl'][0].mean() * 2 > duh['kl'][-1].mean():
                break
        by_model[model] = duh

    # print the outcome
    for model, item in by_model.items():
        if 'MAP' in model:
            print(model, ('\t n0={n0:.0f},'
                          '\t kl={value:.3f}').format(**item))
        else:
            print(model, ('\t alpha={scheduler_exponent},'
                          '\t lr={lr:.1e},'
                          '\t kl={value:.3f}').format(**item))

    return by_model


def curve_plot(bestof, nsteps, figsize, logscale=False, endstep=400, confidence=(5, 95)):
    """Draw mean trajectory plot with percentiles"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for model, item in sorted(bestof.items()):
        xx = item['steps']
        values = item['kl']

        # truncate plot for k-invariance
        end_id = np.searchsorted(xx, endstep) + 1
        xx = xx[:end_id]
        values = values[:end_id]

        # plot mean and percentile statistics
        ax.plot(xx, values.mean(axis=1), label=model,
                marker=MARKERS[model], markevery=len(xx) // 6, markeredgewidth=0,
                color=COLORS[model], alpha=.9)
        ax.fill_between(
            xx,
            np.percentile(values, confidence[0], axis=1),
            np.percentile(values, confidence[1], axis=1),
            alpha=.4,
            color=COLORS[model]
        )

    ax.axvline(nsteps, linestyle='--', color='black')
    ax.grid(True)
    if logscale:
        ax.set_yscale('log')
    ax.set_ylabel('$KL(p^*, p_t)$')
    ax.set_xlabel('number of samples t')
    ax.legend()

    return fig, ax


def scatter_plot(bestof, nsteps, figsize, logscale=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    alldist = []
    allkl = []
    for model, item in sorted(bestof.items()):
        if 'scoredist' not in item:
            continue
        index = min(np.searchsorted(item['steps'], nsteps), len(item['steps']) - 1)

        initial_distances = item['scoredist'][0]
        end_kl = item['kl'][index]
        ax.scatter(
            initial_distances,
            end_kl,
            alpha=.3,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=0,
            label=model if False else None
        )
        alldist += list(initial_distances)
        allkl += list(end_kl)

    # linear regression
    slope, intercept, rval, pval, _ = scipy.stats.linregress(alldist, allkl)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(
        x_vals, y_vals, '--', color='black', alpha=.8,
        label=f'y=ax, r2={rval ** 2:.2f}'
              f',\na={slope:.1e}, b={intercept:.2f}'
    )

    # look
    ax.legend()
    ax.grid(True)
    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(min(alldist), max(alldist))
    else:
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)

    ax.set_ylabel(f'KL(p^*, p_T={nsteps})')
    ax.set_xlabel(r'$||  \theta^{(0)} - \theta^* ||^2$')
    return fig, ax


def two_plots(results, nsteps, plotname, dirname, verbose=False, figsize=(6, 3)):
    print(dirname, plotname)
    bestof = get_best(results, nsteps)
    # remove the models I don't want to compare
    # eg remove SGD, MAP. Keep ASGD and rename them to remove average.
    selected = {
        key[0].capitalize() + key[1:-len('_average')].replace('A', 'X').replace('B', 'Y'): item for
        key, item in bestof.items()
        if key.endswith('_average')}
    if dirname.startswith('guess'):
        selected.pop('Joint', None)

    curves, ax1 = curve_plot(selected, nsteps, figsize, logscale=True)
    # initstring = 'denseinit' if results[0]["is_init_dense"] else 'sparseinit'
    # curves.suptitle(f'Average KL tuned for {nsteps} samples with {confidence} percentiles, '
    #                 f'{initstring},  k={results[0]["k"]}')
    scatter, ax2 = scatter_plot(selected, nsteps, figsize,
                                logscale=dirname == 'guess_sparseinit')

    if verbose:
        for ax in [ax1, ax2]:
            info = str(next(iter(selected.values()))['hyperparameters'])
            txt = ax.text(0.5, 1, info, ha='center', va='top',
                          wrap=True, transform=ax.transAxes,
                          # bbox=dict(boxstyle='square')
                          )
            txt._get_wrap_line_width = lambda: 400.  # wrap to 600 screen pixels

    # small adjustments for intervention guessing
    if dirname.startswith('guess'):
        curves.axes[0].set_ylim(0, 1.5)
        for fig in [curves, scatter]:
            fig.axes[0].set_xlabel('')
            fig.axes[0].set_ylabel('')

    for style, fig in {'curves': curves, 'scatter': scatter}.items():
        for figpath in [os.path.join('plots', dirname, style, f'{style}_{plotname}.pdf')]:
            os.makedirs(os.path.dirname(figpath), exist_ok=True)
            # os.path.join('plots/sweep/png', f'{style}_{plotname}.png')]:
            fig.savefig(figpath, bbox_inches='tight')
    plt.close(curves)
    plt.close(scatter)
    print()


def plot_marginal_likelihoods(results, intervention, k, dirname):
    exp = results[0]
    values = {}
    for whom in ['A', 'B']:
        values[whom] = exp['loglikelihood' + whom][:100].cumsum(0)
        xx = np.arange(1, values[whom].shape[0] + 1)
        values[whom] /= xx[:, np.newaxis]

    if intervention == 'cause':
        right, wrong = 'A', 'B'
    else:
        right, wrong = 'B', 'A'

    plt.plot(values[wrong] - values[right], alpha=.2)
    plt.hlines(0, 0, values['B'].shape[0])
    plt.grid()
    plt.ylim(-1, 1)
    figpath = os.path.join('plots', dirname, 'guessing', f'guess_{intervention}_k={k}.pdf')
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, bbox_inches='tight')
    plt.close()


def merge_results(results1, results2, bs=5):
    """Combine results from intervention on cause and effect.
    Also report statistics about pooled results.

    Pooled records the average over 10 cause and 10 effect interventions
    the goal is to have tighter percentile curves
    which are representative of the algorithm's performance
    """
    combined = []
    pooled = []
    for e1, e2 in zip(results1, results2):
        h1, h2 = e1['hyperparameters'], e2['hyperparameters']
        assert h1['lr'] == h2['lr']
        t1, t2 = e1['trajectory'], e2['trajectory']
        combined_trajs = {'steps': t1['steps']}
        pooled_trajs = combined_trajs.copy()
        for key in t1.keys():
            if key.startswith(('scoredist', 'kl')):
                combined_trajs[key] = np.concatenate((t1[key], t2[key]), axis=1)
                meantraj = (t1[key] + t2[key]) / 2
                pooled_trajs[key] = np.array([
                    meantraj[:, bs * i:bs * (i + 1)].mean(axis=1)
                    for i in range(meantraj.shape[1] // bs)
                ]).T
        combined += [{'hyperparameters': h1, 'trajectory': combined_trajs}]
        pooled += [{'hyperparameters': h2, 'trajectory': pooled_trajs}]
    return combined, pooled


def all_plot(guess, dense, results_dir='results', figsize=(5, 2.5)):
    basefile = '_'.join(['guess' if guess else 'sweep2',
                         'denseinit' if dense else 'sparseinit'])
    print(basefile, '\n---------------------')

    for k in [10, 20, 50]:
        # Optimize hyperparameters for nsteps such that curves are k-invariant
        nsteps = k ** 2 // 4
        allresults = defaultdict(list)
        for intervention in ['cause', 'effect']:
            # , 'gmechanism', 'independent', 'geometric', 'weightedgeo']:
            plotname = f'{intervention}_k={k}'
            file = basefile + '_' + plotname + '.pkl'
            filepath = os.path.join(results_dir, file)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as fin:
                    results = pickle.load(fin)
                    two_plots(results, nsteps, plotname=plotname, dirname=basefile,
                              figsize=figsize)
                    allresults[intervention] = results
                    if guess:
                        pass
                        # plot_marginal_likelihoods(results, intervention, k, basefile)

        if not guess and 'cause' in allresults and 'effect' in allresults:
            combined, pooled = merge_results(allresults['cause'], allresults['effect'])
            if len(combined) > 0:
                for key, item in {'combined': combined, 'pooled': pooled}.items():
                    two_plots(item, nsteps, plotname=f'{key}_k={k}', dirname=basefile,
                              figsize=figsize)


if __name__ == '__main__':
    # all_plot(guess=True, dense=True)
    # all_plot(guess=True, dense=False)
    all_plot(guess=False, dense=True)
    all_plot(guess=False, dense=False)
