import torch.nn as nn
from ops import OPS


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for primitive in OPS.keys():
            op = OPS[primitive](C, stride, affine=False)
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))
