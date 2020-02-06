import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)

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
COLORS = {**COLORS, **{key[0].capitalize() + key[1:]: item for key, item in COLORS.items()}}


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
    # idea : store per model each parameter and kl values
    # then for each model return the argmax parameters and curves
    # for kl and integral kl

    by_model = {}
    for exp in results:
        relevant_parameters = {
            key: exp[key] for key in exp
            if key in ['k', 'lr', 'n0', 'scheduler_exponent', 'steps']}

        for model, metric in value_at_step(exp, nsteps).items():
            if model not in by_model:
                by_model[model] = []
            toadd = {**relevant_parameters,
                     'value': metric,
                     'kl': exp['kl_' + model]
                     }
            if 'scoredist_' + model in exp:
                toadd['scoredist'] = exp['scoredist_' + model]
            by_model[model] += [toadd]

    for model, metrics in by_model.items():
        by_model[model] = sorted(metrics, key=lambda x: x['value'])[0]

    for model, item in by_model.items():
        if 'MAP' in model:
            print(model, ('\t n0={n0:.0f},'
                          '\t kl={value:.3f}').format(**item))

    for model, item in by_model.items():
        if 'MAP' not in model:
            print(model, ('\t alpha={scheduler_exponent},'
                          '\t lr={lr:.1e},'
                          '\t kl={value:.3f}').format(**item))

    return by_model


def curve_plot(bestof, nsteps, figsize, confidence=(5, 95), logscale=False):
    """Draw mean trajectory plot with percentiles"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for model, item in sorted(bestof.items()):
        xx = item['steps']
        values = item['kl']

        # truncate plot for k-invariance
        k = item['k']
        end = np.searchsorted(xx, 2 * nsteps) + 1
        xx = xx[:end]
        values = values[:end]

        # plot mean and percentile statistics
        ax.plot(xx, values.mean(axis=1), label=model, color=COLORS[model], alpha=.9)
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

    return fig


def scatter_plot(bestof, nsteps, figsize, logscale=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    alldist = []
    for model, item in sorted(bestof.items()):
        if 'scoredist' not in item:
            continue
        index = min(np.searchsorted(item['steps'], nsteps), len(item['steps'])-1)

        initial_distances = item['scoredist'][0]
        end_kl = item['kl'][index]
        ax.scatter(
            initial_distances,
            end_kl,
            alpha=.3,
            color=COLORS[model],
            label=model
        )
        alldist += list(initial_distances)

    ax.legend()
    ax.grid(True)
    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(min(alldist), max(alldist))
    else:
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)

    ax.set_ylabel(f'KL(p^*, p_T={nsteps})')
    # ax.set_xlabel('||transfer - model||^2 at initialization')
    ax.set_xlabel(r'$|| \mathbf{s_0 - s^*} ||^2$')
    return fig


def two_plots(results, nsteps, plotname, dirname):
    print(dirname, plotname)
    bestof = get_best(results, nsteps)
    # remove the models I don't want to compare
    # eg remove SGD, MAP. Keep ASGD and rename them to remove average.
    selected = {key[0].capitalize() + key[1:-len('_average')].replace('A', 'X').replace('B','Y'): item for key, item in bestof.items()
                if key.endswith('_average')}
    if dirname.startswith('guess'):
        selected.pop('Joint', None)

    figsize = (6, 3)
    confidence = (5, 95)
    curves = curve_plot(selected, nsteps, figsize, confidence)
    # initstring = 'denseinit' if results[0]["is_init_dense"] else 'sparseinit'
    # curves.suptitle(f'Average KL tuned for {nsteps} samples with {confidence} percentiles, '
    #                 f'{initstring},  k={results[0]["k"]}')

    scatter = scatter_plot(selected, nsteps, figsize,
                           logscale=dirname == 'guess_sparseinit')

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


def all_plot(guess, dense, results_dir='results', nsteps=None):
    basefile = '_'.join(['guess' if guess else 'sweep2',
                         'denseinit' if dense else 'sparseinit'])
    print(basefile, '\n---------------------')

    for k in [10, 20, 50]:
        # Optimize hyperparameters for nsteps such that curves are k-invariant
        if nsteps is None:
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
                    two_plots(results, nsteps, plotname=plotname, dirname=basefile)
                    allresults[intervention] = results
                    if guess:
                        pass
                        # plot_marginal_likelihoods(results, intervention, k, basefile)

        if not guess and 'cause' in allresults and 'effect' in allresults:
            # now let's combine results from intervention on cause and effect
            # let's also report statistics about pooled results
            # do not do it if we guessed the intervention because it's not straightforward
            combineds = []
            pooleds = []
            for e1, e2 in zip(allresults['cause'], allresults['effect']):
                assert e1['lr'] == e2['lr']
                combined = e1.copy()
                pooled = e1.copy()
                for key in e2.keys():
                    if key.startswith(('scoredist', 'kl')):
                        combined[key] = np.concatenate((e1[key], e2[key]), axis=1)
                        # pooled records the average over 10 cause and 10 effect interventions
                        # the goal is to have tighter percentile curves
                        # which are representative of the algorithm's performance
                        meantraj = (e1[key] + e2[key]) / 2
                        bs = 5
                        pooled[key] = np.array([meantraj[:, bs * i:bs * (i + 1)].mean(axis=1)
                                                for i in range(meantraj.shape[1] // bs)]).T
                combineds += [combined]
                pooleds += [pooled]
            if len(combineds) > 0:
                two_plots(combineds, nsteps, plotname=f'combined_k={k}', dirname=basefile)
                two_plots(pooleds, nsteps, plotname=f'pooled_k={k}', dirname=basefile)


if __name__ == '__main__':
    all_plot(guess=True, dense=True)
    all_plot(guess=True, dense=False)
    # all_plot(guess=False, dense=True)
    # all_plot(guess=False, dense=False)
