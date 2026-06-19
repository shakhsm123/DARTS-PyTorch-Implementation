import torch.nn.functional as F
from collections import namedtuple
from ops import OPS

Genotype = namedtuple('Genotype', ['normal', 'normal_concat', 'reduce', 'reduce_concat'])


def get_genotype(model):
    op_names = list(OPS.keys())

    def parse(alpha):
        softmax_alpha = F.softmax(alpha, dim=-1)
        gene = []
        offset = 0
        for node in [2, 3, 4, 5]:
            num_edges = node
            best_ops = []
            for j in range(num_edges):
                row = softmax_alpha[offset + j]
                best_op_idx = row[1:].argmax().item() + 1
                best_op_strength = row[1:].max().item()
                best_ops.append((best_op_strength, op_names[best_op_idx], j))
            best_ops.sort(reverse=True)
            for strength, op_name, source in best_ops[:2]:
                gene.append((op_name, source))
            offset += num_edges
        return gene

    return Genotype(
        normal=parse(model.alpha_normal),
        normal_concat=[2, 3, 4, 5],
        reduce=parse(model.alpha_reduce),
        reduce_concat=[2, 3, 4, 5]
    )
