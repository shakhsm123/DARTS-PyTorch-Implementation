import torch
import torch.nn as nn
from ops import ReLUConvBN, FactorizedReduce
from mixed_op import MixedOp


class Cell(nn.Module):
    def __init__(self, steps, C_prev, C_prev_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess_conv2d = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess_conv2d = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess_conv2d_2 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.steps = steps
        self.edges = nn.ModuleList()
        for i in range(steps):
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self.edges.append(op)

    def forward(self, x_s0, x_s1, weights):
        x_s0 = self.preprocess_conv2d(x_s0)
        x_s1 = self.preprocess_conv2d_2(x_s1)
        states = [x_s0, x_s1]
        offset = 0
        for i in range(self.steps):
            s = sum(
                self.edges[offset + j](states[j], weights[offset + j])
                for j in range(len(states))
            )
            offset += len(states)
            states.append(s)
        return torch.cat(states[2:], dim=1)
