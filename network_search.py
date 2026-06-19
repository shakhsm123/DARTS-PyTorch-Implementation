import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_search import Cell


class Search_Network(nn.Module):
    def __init__(self, C_init, num_classes, total_cells=8, steps=4):
        super().__init__()
        C_current = C_init
        C_prev = C_init * 3
        C_prev_prev = C_init * 3
        reduction_prev = False
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_init * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_init * 3)
        )
        self.cells = nn.ModuleList()
        for i in range(total_cells):
            reduction = (i == total_cells // 3 or i == 2 * total_cells // 3)
            cell = Cell(steps, C_prev, C_prev_prev, C_current, reduction, reduction_prev)
            self.cells.append(cell)
            C_prev_prev = C_prev
            C_prev = C_current * 4
            if reduction:
                C_current *= 2
            reduction_prev = reduction
        self.alpha_normal = nn.Parameter(torch.randn(14, 8))
        self.alpha_reduce = nn.Parameter(torch.randn(14, 8))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def arch_parameters(self):
        return self.alpha_normal, self.alpha_reduce

    def forward(self, x):
        weights_normal = F.softmax(self.alpha_normal, dim=-1)
        weights_reduce = F.softmax(self.alpha_reduce, dim=-1)
        s1 = s0 = self.stem(x)
        for cell in self.cells:
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1, weights_reduce)
            else:
                s0, s1 = s1, cell(s0, s1, weights_normal)
        s1 = self.global_pooling(s1)
        s1 = s1.view(s1.size(0), -1)
        logits = self.classifier(s1)
        return logits
