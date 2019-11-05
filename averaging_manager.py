from collections import defaultdict

import torch
from torch import nn, optim


class AveragedModel:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.current_parameters = {}

    def __enter__(self):
        """Assign average value to self.model parameters"""
        for p in self.model.parameters():
            self.current_parameters[p] = p.data
            if p in self.optimizer.state:
                p.data = self.optimizer.state[p]['ax']
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.model.parameters():
            if p in self.current_parameters:
                p.data = self.current_parameters[p]


def test_AveragedModel(d=10):
    torch.manual_seed(1)
    model = nn.Linear(d, 1, bias=False)
    optimizer = optim.ASGD(model.parameters(), lr=.05,
                           lambd=0, alpha=0, t0=0, weight_decay=0)

    print(next(model.parameters()))
    xeval = torch.randn(100, d)
    yeval = xeval.mean(dim=1, keepdim=True)

    trajectory = defaultdict(list)
    for t in range(100):
        # evaluate
        with torch.no_grad():
            trajectory['error'] += [float(torch.mean((model(xeval) - yeval) ** 2))]
            trajectory['pdist'] += [float(torch.norm((next(model.parameters()).data
                                                      - 1 / d * torch.ones(d)) ** 2))]
            with AveragedModel(model, optimizer):
                trajectory['aerror'] += [float(torch.mean((model(xeval) - yeval) ** 2))]
                trajectory['apdist'] += [float(torch.norm((next(model.parameters()).data
                                                           - 1 / d * torch.ones(d)) ** 2))]

        # train
        optimizer.zero_grad()
        x = torch.randn(1, d)
        y = torch.mean(x, dim=1, keepdim=True)
        loss = torch.mean((y - model(x)) ** 2)
        loss.backward()
        optimizer.step()

    print(next(model.parameters()).data)

    import matplotlib.pyplot as plt
    # plt.scatter(error, aerror, c=np.arange(len(error)))
    plt.plot(trajectory['error'], alpha=.5, label='MSE')
    plt.plot(trajectory['aerror'], alpha=.5, label='average MSE')
    plt.plot(trajectory['pdist'], alpha=.5, label='pdist')
    plt.plot(trajectory['apdist'], alpha=.5, label='average pdist')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_AveragedModel()
