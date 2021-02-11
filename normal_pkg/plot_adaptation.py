import os
import pickle
import matplotlib
from categorical.plot_sweep import two_plots
from adaptation import CholeskyModule

CholeskyModule

def learning_curves(results_dir='normal_results'):
    for k in [10]:
        # Optimize hyperparameters for nsteps
        nsteps = 100
        init = 'natural'
        for intervention in ['cause', 'effect', 'mechanism']:
            plotname = f'{intervention}_{init}_k={k}'
            filepath = os.path.join(results_dir, plotname + '.pkl')
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as fin:
                    results = pickle.load(fin)
                    two_plots(results, nsteps, plotname=plotname, dirname='normal_adaptation',
                              figsize=(3, 3))


if __name__ == "__main__":
    matplotlib.use('pgf')
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['pdf.fonttype'] = 42
    learning_curves()
