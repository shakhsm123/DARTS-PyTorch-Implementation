import torch.nn as nn
from cell_eval import Cell_Eval


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Network_Eval(nn.Module):
    def __init__(self, C_init, num_classes, genotype, total_cells=20, steps=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_init * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_init * 3)
        )
        self.cells = nn.ModuleList()
        C_curr = C_init
        C_prev = C_init * 3
        C_prev_prev = C_init * 3
        reduction_prev = False
        aux_C = C_init * 3
        self.aux_position = -1
        for i in range(total_cells):
            reduction = (i == total_cells // 3 or i == 2 * total_cells // 3)
            gene = genotype.reduce if reduction else genotype.normal
            if i == 2 * total_cells // 3:
                self.aux_position = i
                aux_C = C_prev
            cell = Cell_Eval(gene, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            C_prev_prev = C_prev
            C_prev = 4 * C_curr
            if reduction:
                C_curr *= 2
            reduction_prev = reduction
        self.auxiliary_head = AuxiliaryHead(aux_C, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_position and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits, aux_logits
