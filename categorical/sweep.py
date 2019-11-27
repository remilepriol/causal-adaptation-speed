import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from categorical.script_plot import COLORS

np.set_printoptions(precision=2)


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
            key: exp[key] for key in
            ['k', 'lr', 'n0', 'scheduler_exponent', 'steps']}

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


def curve_plot(bestof, figsize, skiplist, confidence=(5, 95)):
    """Draw mean trajectory plot with percentiles"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for model, item in bestof.items():
        # discard non averaged models
        if model in skiplist:
            continue

        xx = item['steps']
        values = item['kl']

        # truncate plot for k-invariance
        k = item['k']
        end = np.searchsorted(xx, k ** 2) + 1
        xx = xx[:end]
        values = values[:end]

        # plot mean and percentile statistics
        ax.plot(xx, values.mean(axis=1), label=model, color=COLORS[model])
        ax.fill_between(
            xx,
            np.percentile(values, confidence[0], axis=1),
            np.percentile(values, confidence[1], axis=1),
            alpha=.4,
            color=COLORS[model]
        )

    ax.grid(True)
    # ax.set_yscale('log')
    ax.set_ylabel('$KL(p^*, p_t)$')
    ax.set_xlabel('number of examples t')
    ax.legend()

    return fig


def scatter_plot(bestof, nsteps, figsize, skiplist):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for model, item in bestof.items():
        if model in skiplist:
            continue
        if 'scoredist' not in item:
            continue
        index = np.searchsorted(item['steps'], nsteps)

        initial_distances = item['scoredist'][0]
        end_kl = item['kl'][index]
        ax.scatter(
            initial_distances,
            end_kl,
            alpha=.3,
            color=COLORS[model],
            label=model
        )

    ax.grid(True)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(f'KL(p^*, p_T={nsteps})')
    ax.legend()
    # ax.set_xlabel('||transfer - model||^2 at initialization')
    ax.set_xlabel(r'$|| \mathbf{s_0 - s^*} ||^2$')
    return fig


def two_plots(results, nsteps, plotname, dirname):
    print()
    print(plotname)
    bestof = get_best(results, nsteps)
    figsize = (6, 3)
    skiplist = ['causal', 'anti', 'joint']  # , 'MAP_uniform', 'MAP_source']
    confidence = (5, 95)
    curves = curve_plot(bestof, figsize, skiplist, confidence)
    initstring = 'syminit' if results[0]["is_init_symmetric"] else 'asyminit'
    curves.suptitle(f'Average KL tuned for {nsteps} samples with {confidence} percentiles, '
                    f'{initstring},  k={results[0]["k"]}')
    scatter = scatter_plot(bestof, nsteps, figsize, skiplist)
    for style, fig in {'curves': curves, 'scatter': scatter}.items():
        for figpath in [os.path.join('plots', dirname, style, f'{plotname}.pdf')]:
            os.makedirs(os.path.dirname(figpath), exist_ok=True)
            # os.path.join('plots/sweep/png', f'{style}_{plotname}.png')]:
            fig.savefig(figpath, bbox_inches='tight')
    plt.close(curves)
    plt.close(scatter)
    print()


def all_plot():
    results_dir = 'results'

    # basefile = 'asyminter_asyminit_parameter_sweep_'
    # for k in [10, 50, 100]:

    for init in ['syminit_', 'asyminit_']:
        basefile = 'sweep2_' + init
        for k in [10, 20, 50]:
            nsteps = k ** 2 // 4
            allresults = defaultdict(list)
            for intervention in ['cause', 'effect', 'independent', 'geometric']:
                # , 'weightedgeo']:
                plotname = f'{intervention}_k={k}'
                file = basefile + plotname + '.pkl'
                with open(os.path.join(results_dir, file), 'rb') as fin:
                    results = pickle.load(fin)
                    # Optimize hyperparameters for nsteps such that curves are k-invariant
                    two_plots(results, nsteps, plotname=plotname,
                              dirname=basefile[:-1])
                    allresults[intervention] = results

            # now let's combine results from intervention on cause and effect
            # let's also report statistics about pooled results
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
                two_plots(combineds, nsteps, plotname=f'combined_k={k}', dirname=basefile[:-1])
                two_plots(pooleds, nsteps, plotname=f'pooled_k={k}', dirname=basefile[:-1])


if __name__ == '__main__':
    all_plot()
