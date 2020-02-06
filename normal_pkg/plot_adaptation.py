import os
import pickle
from collections import defaultdict

from categorical.sweep import two_plots
from normal_pkg.adaptation import CholeskyModule


def learning_curves(results_dir='normal_results'):
    for k in [3, 4, 10]:
        # Optimize hyperparameters for nsteps such that curves are k-invariant
        nsteps = 100
        allresults = defaultdict(list)
        init = 'natural'
        for intervention in ['cause', 'effect']:
            plotname = f'{intervention}_{init}_k={k}'
            filepath = os.path.join(results_dir, plotname + '.pkl')
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as fin:
                    results = pickle.load(fin)
                    two_plots(results, nsteps, plotname=plotname, dirname='normal')
                    allresults[intervention] = results


if __name__ == "__main__":
    learning_curves()
