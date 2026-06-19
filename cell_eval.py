import torch
import torch.nn as nn
from ops import OPS, ReLUConvBN, FactorizedReduce


def drop_path(x, drop_prob):
    if drop_prob > 0 and torch.is_grad_enabled():
        keep_prob = 1 - drop_prob
        mask = torch.zeros(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = x / keep_prob * mask
    return x


class Cell_Eval(nn.Module):
    def __init__(self, gene, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        if reduction_prev:
            self.preprocess_conv2d = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess_conv2d = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
        self.ops = nn.ModuleList()
        self.sources = []
        for op_name, source in gene:
            stride = 2 if reduction and source < 2 else 1
            op = OPS[op_name](C, stride, affine=True)
            self.sources.append(source)
            self.ops.append(op)
        self.steps = 4
        self.drop_prob = 0.0

    def forward(self, s0, s1):
        s0 = self.preprocess_conv2d(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self.steps):
            h1 = drop_path(self.ops[2 * i](states[self.sources[2 * i]]), self.drop_prob)
            h2 = drop_path(self.ops[2 * i + 1](states[self.sources[2 * i + 1]]), self.drop_prob)
            states.append(h1 + h2)
        return torch.cat(states[2:], dim=1)
