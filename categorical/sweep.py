import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from categorical.script_plot import COLORS

np.set_printoptions(precision=2)


def integral_kl(trajectory, nsteps=1000):
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
            ['lr', 'n0', 'scheduler_exponent', 'steps']}

        for model, metric in integral_kl(exp, nsteps).items():
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
        if not 'MAP' in model:
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
    ax.set_ylabel(r'$KL(p^*, p_{T=1000})$')
    ax.legend()
    # ax.set_xlabel('||transfer - model||^2 at initialization')
    ax.set_xlabel(r'$|| \mathbf{s_0 - s^*} ||^2$')
    return fig


def two_plots(results, nsteps, plotname):
    print()
    print(plotname)
    bestof = get_best(results, nsteps)
    figsize = (6, 3)
    skiplist = ['causal', 'anti', 'joint']  # , 'MAP_uniform', 'MAP_source']
    curves = curve_plot(bestof, figsize, skiplist)
    scatter = scatter_plot(bestof, nsteps, figsize, skiplist)
    os.makedirs('plots/sweep/png', exist_ok=True)
    for style, fig in {'curves': curves, 'scatter': scatter}.items():
        for figpath in [os.path.join('plots/sweep', f'{style}_{plotname}.pdf')]:
            # os.path.join('plots/sweep/png', f'{style}_{plotname}.png')]:
            fig.savefig(figpath, bbox_inches='tight')
    plt.close(curves)
    plt.close(scatter)
    print()


def all_plot():
    results_dir = 'results'
    basefile = 'asyminter_asyminit_parameter_sweep_'
    for k in [10, 50, 100]:
        rr = defaultdict(list)
        for intervention in ['independent', 'cause', 'effect', 'geometric', 'weightedgeo']:
            plotname = f'{intervention}_k={k}'
            file = basefile + plotname + '.pkl'
            with open(os.path.join(results_dir, file), 'rb') as fin:
                results = pickle.load(fin)
                two_plots(results, nsteps=400, plotname=plotname)
                rr[intervention] = results

        # now let's combine results from intervention on cause and effect
        rrr = []
        for e1, e2 in zip(rr['cause'], rr['effect']):
            assert e1['lr'] == e2['lr']
            combined = e1.copy()
            for key in e2.keys():
                if key.startswith(('scoredist', 'kl')):
                    combined[key] = np.concatenate((e1[key], e2[key]), axis=1)
            rrr += [combined]
        if len(rrr) > 0:
            two_plots(rrr, nsteps=400, plotname=f'combined_k={k}')


if __name__ == '__main__':
    all_plot()
