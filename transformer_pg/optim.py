import torch


class Noam:
    "Optimizer wrapper that implements rate."

    def __init__(self, model_size, factor, warm_up, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warm_up = warm_up
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warm_up ** (-1.5)))


def get_std_opt(model):
    return Noam(model.d_model, 2, 16000,
                torch.optim.Adam(model.parameters(), lr=0.2, betas=(0.9, 0.98), eps=1e-9))