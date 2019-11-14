import numpy as np

from categorical.distances1 import sample_joint


# Goal have one reproducible for loop that takes one model as argument

# update rule, meaning optimization hyperparameters
# are set in each model


class Experiment:

    def __init__(self, k, n, intervention, is_init_symmetric,
                 is_intervention_symmetric, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.causalstatic = sample_joint(k, n, concentration=1,
                                         is_init_symmetric=is_init_symmetric)
        self.transferstatic = self.causalstatic.intervention(
            on=intervention,
            concentration=1,
            fromjoint=is_intervention_symmetric
        )
        self.transfer = None
        self.results = {}

    def train(self, model, batch_size=1, max_iter=100):
        # set random seed
        # sample distributions
        np.random.seed(self.seed)
        kl = [self.transfer.kullback_leibler(model)]
        for t in range(max_iter):
            samples = self.transfer.sample(batch_size)
            model.update(samples)
            kl += [self.transfer.kullback_leibler(model.to_joint())]

        self.results[model.name] = kl




if __name__ == "__main__":
    print('hi')
    models = {}
    for direction in ['independent', 'joint', 'causal', 'anti']:
        for rule in ['ASGD', 'MAP']:
            model_name = direction + '_' + rule
            models[model_name] = []

    print(models)
