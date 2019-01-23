from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

def pdist(A, squared = False, eps = 1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res if squared else res.clamp(min = eps).sqrt()

class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, embeddings, labels):
        '''
        score = embeddings
        target = labels
        loss = 0
        counter = 0
        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        
        for i in range(bsz):
            t_i = target[i].item()
            for j in range(i + 1, bsz):
                t_j = target[j].item()
                if t_i == t_j:
                    # Negative component
                    # !! Could do other things (like softmax that weights closer negatives)
                    l_ni = (self.margin - dist[i][target != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][target != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    # Positive component
                    l_p  = dist[i,j]
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1
        return loss / (2 * counter), 0
        '''
        margin = 1.0
        eps = 1e-4
        d = pdist(embeddings, squared = False, eps = eps)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
        neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
        return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d)), 0

def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(LiftedStructureLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


