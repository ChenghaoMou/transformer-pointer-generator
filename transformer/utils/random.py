import torch
from torch.autograd import Variable
import numpy as np


def data_gen(V, batch, nbatches, device='cuda'):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False).to(device).transpose(0, 1)
        tgt = Variable(data, requires_grad=False).to(device).transpose(0, 1)
        yield Batch(src, tgt, 0)


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        if trg is not None:
            self.trg = trg
