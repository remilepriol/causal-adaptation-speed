import math

import torch
from torch import optim


class PerturbedProximalGradient(optim.ASGD):

    def __init__(self, params, **kwargs):
        super(PerturbedProximalGradient, self).__init__(params, **kwargs)

    def step(self):
        """Mostly copied from ASGD, but there was no other way to do both
         projection, proximal step and averaging.
         """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # decay term
                p.data.mul_(1 - group['lambd'] * state['eta'])

                # update parameter
                p.data.add_(-state['eta'], grad)

                # NEW
                # project back onto lower triangular matrices
                p.data = torch.tril(p.data)
                # proximal update on diagonal parameters with - log loss
                olddiag = torch.diag(p.data)
                newdiag = .5 * (olddiag + torch.sqrt(olddiag ** 2 + state['eta']))
                mask = torch.eye(olddiag.shape[0])
                p.data = (1 - mask) * p.data + torch.diag(newdiag)
                # END NEW

                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)

                # update eta and mu
                state['eta'] = (group['lr'] /
                                math.pow((1 + group['lambd'] * group['lr'] * state['step']),
                                         group['alpha']))
                state['mu'] = 1 / max(1, state['step'] - group['t0'])