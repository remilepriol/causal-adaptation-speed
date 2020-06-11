import argparse

import categorical
from categorical import script_experiments,plot_sweep,plot_old
from normal_pkg import adaptation, distances, plot_adaptation, plot_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate the speed of adpatation of cause-effect models.')
    parser.add_argument('distribution', type=str, choices=['categorical', 'normal'])
    parser.add_argument('action', type=str, choices=['distance', 'adaptation', 'plot'])

    args = parser.parse_args()

    if args.distribution == 'categorical':
        if args.action == 'distance':
            categorical.script_experiments.all_distances()
            categorical.plot_old.dist_plot()

        elif args.action == 'adaptation':
            for init_dense in [True, False]:
                for k in [20]:
                    for intervention in ['cause', 'effect', 'singlecond']:
                        categorical.script_experiments.parameter_sweep(intervention, k, init_dense)

        elif args.action == 'plot':
            for dense in [True, False]:
                categorical.plot_sweep.all_plot(dense=dense, results_dir='categorical_results')

    elif args.distribution == 'normal':
        if args.action == 'distance':
            print("Measuring distances")
            distances.record_distances()
            plot_distances.all_distances()

        elif args.action == 'adaptation':
            print("\n\n Simulating adaptation to interventions")
            base_hparams = {'n': 100, 'T': 400, 'batch_size': 1, 'use_prox': True,
                            'log_interval': 10, 'intervention_scale': 1,
                            'init': 'natural', 'preccond_scale': 10}
            print("Hyperparameters ", base_hparams)
            lrlr = [.001, .003, .01, .03, .1]
            for k in [10]:
                base_hparams['k'] = k
                for intervention in ['cause', 'effect']:
                    hparams = {**base_hparams, 'k': k, 'intervention': intervention}
                    adaptation.sweep_lr(lrlr, hparams)

        elif args.action == 'plot':
            plot_adaptation.learning_curves()
