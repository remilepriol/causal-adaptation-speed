from collections import defaultdict

import numpy as np
import torch
import tqdm
from torch import optim

from averaging_manager import AveragedModel
from categorical.models import CategoricalModule, Counter, JointMAP, JointModule, sample_joint


def experiment_optimize(k, n, T, lr, intervention,
                        concentration=1,
                        is_init_dense=False,
                        batch_size=10, scheduler_exponent=0, n0=10,
                        log_interval=10, use_map=False):
    """Measure optimization speed and parameters distance.

    Hypothesis: initial distance to optimum is correlated to optimization speed with SGD.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Use SGD to update a causal
    and an anticausal model for T steps. At each step, measure KL
    and distance in scores for causal and anticausal directions.
    """
    causalstatic = sample_joint(k, n, concentration, is_init_dense)
    transferstatic = causalstatic.intervention(
        on=intervention,
        concentration=concentration,
        dense=is_init_dense
    )
    # MODULES
    causal = causalstatic.to_module()
    transfer = transferstatic.to_module()

    anticausal = causalstatic.reverse().to_module()
    antitransfer = transferstatic.reverse().to_module()

    joint = JointModule(causal.to_joint().detach().view(n, -1))
    jointtransfer = JointModule(transfer.to_joint().detach().view(n, -1))

    # Optimizers
    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0,
                 'weight_decay': 0}
    causaloptimizer = optim.ASGD(causal.parameters(), **optkwargs)
    antioptimizer = optim.ASGD(anticausal.parameters(), **optkwargs)
    jointoptimizer = optim.ASGD(joint.parameters(), **optkwargs)
    optimizers = [causaloptimizer, antioptimizer, jointoptimizer]

    # MAP
    counter = Counter(np.zeros([n, k, k]))
    smooth_MLE = JointMAP(np.ones([n, k, k]), counter)
    joint_MAP = JointMAP(n0 * causalstatic.to_joint(return_probas=True), counter)

    steps = []
    ans = defaultdict(list)
    for t in tqdm.tqdm(range(T)):

        # EVALUATION
        if t % log_interval == 0:
            steps.append(t)

            with torch.no_grad():

                for model, optimizer, target, name in zip(
                        [causal, anticausal, joint],
                        optimizers,
                        [transfer, antitransfer, jointtransfer],
                        ['causal', 'anti', 'joint']
                ):
                    # SGD
                    ans[f'kl_{name}'].append(target.kullback_leibler(model))
                    ans[f'scoredist_{name}'].append(target.scoredist(model))

                    # ASGD
                    with AveragedModel(model, optimizer) as m:
                        ans[f'kl_{name}_average'].append(
                            target.kullback_leibler(m))
                        ans[f'scoredist_{name}_average'].append(
                            target.scoredist(m))

                # MAP
                if use_map:
                    ans['kl_MAP_uniform'].append(
                        transfer.kullback_leibler(smooth_MLE))
                    ans['kl_MAP_source'].append(
                        transfer.kullback_leibler(joint_MAP))

        # UPDATE
        for opt in optimizers:
            opt.lr = lr / t ** scheduler_exponent
            opt.zero_grad()

        if batch_size == 'full':
            causalloss = transfer.kullback_leibler(causal).sum()
            antiloss = antitransfer.kullback_leibler(anticausal).sum()
            jointloss = jointtransfer.kullback_leibler(joint).sum()
        else:
            aa, bb = transferstatic.sample(m=batch_size)
            if use_map:
                counter.update(aa, bb)
            taa, tbb = torch.from_numpy(aa), torch.from_numpy(bb)
            causalloss = - causal(taa, tbb).sum() / batch_size
            antiloss = - anticausal(taa, tbb).sum() / batch_size
            jointloss = - joint(taa, tbb).sum() / batch_size

        for loss, opt in zip([causalloss, antiloss, jointloss], optimizers):
            loss.backward()
            opt.step()

    for key, item in ans.items():
        ans[key] = torch.stack(item).numpy()

    return {'steps': np.array(steps), **ans}


def test_experiment_optimize():
    for intervention in ['cause', 'effect', 'gmechanism']:
        experiment_optimize(
            k=2, n=3, T=6, lr=.1, batch_size=4, log_interval=1,
            intervention=intervention, use_map=True
        )


def experiment_guess(
        k, n, T, lr, intervention,
        concentration=1, is_init_dense=False,
        batch_size=10, scheduler_exponent=0, log_interval=10
):
    """Measure optimization speed after guessing intervention.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Initialize a causal and
    an anticausal model to the reference distribution. Duplicate them
    and initialize one module of each duplicata to the uniform.
    Run the optimization and record KL and distance. Also record the
    accuracy of guessing the intervention based on the lowest KL.
    """
    causalstatic = sample_joint(k, n, concentration, is_init_dense)
    transferstatic = causalstatic.intervention(
        on=intervention,
        concentration=concentration,
        dense=is_init_dense
    )
    causal = causalstatic.to_module()
    transfer = transferstatic.to_module()

    anticausal = causalstatic.reverse().to_module()
    antitransfer = transferstatic.reverse().to_module()

    joint = JointModule(causal.to_joint().detach().view(n, -1))
    jointtransfer = JointModule(transfer.to_joint().detach().view(n, -1))

    # TODO put all models and their transfer into a dict
    models = [causal, anticausal, joint]
    names = ['causal', 'anti', 'joint']
    targets = [transfer, antitransfer, jointtransfer]

    # step 1 : duplicate models with intervention guessing
    causalguessA = CategoricalModule(torch.zeros([n, k]), causal.sba, is_btoa=False)
    causalguessB = CategoricalModule(causal.sa, torch.zeros([n, k, k]), is_btoa=False)

    antiguessA = CategoricalModule(anticausal.sa, torch.zeros([n, k, k]), is_btoa=True)
    antiguessB = CategoricalModule(torch.zeros([n, k]), anticausal.sba, is_btoa=True)

    models += [causalguessA, causalguessB, antiguessA, antiguessB]
    names += ['CausalGuessA', 'CausalGuessB', 'AntiGuessA', 'AntiGuessB']
    targets += [transfer, transfer, antitransfer, antitransfer]

    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0, 'weight_decay': 0}
    optimizer = optim.ASGD([p for m in models for p in m.parameters()], **optkwargs)

    # intervention guess
    marginalA = JointModule(causal.sa)
    marginalB = JointModule(anticausal.sa)

    steps = []
    ans = defaultdict(list)
    for step in tqdm.tqdm(range(T)):

        # EVALUATION
        if step % log_interval == 0:
            steps.append(step)
            with torch.no_grad():

                for model, target, name in zip(models, targets, names):
                    # SGD
                    ans[f'kl_{name}'].append(target.kullback_leibler(model))
                    ans[f'scoredist_{name}'].append(target.scoredist(model))

                    # ASGD
                    with AveragedModel(model, optimizer):
                        ans[f'kl_{name}_average'].append(target.kullback_leibler(model))
                        ans[f'scoredist_{name}_average'].append(target.scoredist(model))

        # UPDATE
        optimizer.lr = lr / step ** scheduler_exponent
        optimizer.zero_grad()

        if batch_size == 'full':
            loss = sum([t.kullback_leibler(m).sum() for m, t in zip(models, targets)])
        else:
            aa, bb = transferstatic.sample(m=batch_size, return_tensor=True)
            loss = sum([- m(aa, bb).sum() for m in models]) / batch_size

            # step 2, estimate likelihood of samples aa and bb for each marginals
            # of the reference model and take the lowest likelihood as a guess
            # for the intervention. Take the average over all examples seen until now
            with torch.no_grad():
                ans['loglikelihoodA'].append(marginalA(torch.zeros_like(aa), aa).mean(dim=1))
                ans['loglikelihoodB'].append(marginalB(torch.zeros_like(bb), bb).mean(dim=1))

        loss.backward()
        optimizer.step()

    for key, item in ans.items():
        ans[key] = torch.stack(item).numpy()

    return {'steps': np.array(steps), **ans}


def test_experiment_guess():
    for bs in ['full', 4]:
        for intervention in ['cause', 'effect']:
            experiment_guess(
                k=5, n=10, T=6, lr=.1, batch_size=bs, log_interval=1,
                intervention=intervention
            )


if __name__ == "__main__":
    test_experiment_optimize()
    test_experiment_guess()
