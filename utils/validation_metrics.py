# encoding: utf-8
def accuracy(score, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = score.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret
