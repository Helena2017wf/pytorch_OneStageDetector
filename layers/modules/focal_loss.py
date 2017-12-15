
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        ### for softmax version
        # pt = F.softmax(input,dim=-1)
        # pt = pt.clamp(self.eps, 1. - self.eps)
        # CE_loss = -1*y*torch.log(pt) # cross entropy

        #### for sigmoid version
        p = F.sigmoid(input)
        pt = p*y + (1-p)*(1-y)
        pt = pt.clamp(self.eps,1. - self.eps)
        CE_loss = -1* torch.log(pt)  ## cross entropy with sigmoid

        loss = CE_loss * self.alpha * (1 - pt) ** self.gamma # focal loss

        return loss.sum()
if __name__ == "__main__":

    FL = FocalLoss(gamma=0)
    N = 4
    C = 2
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)
    ce_loss = F.cross_entropy(inputs_ce, targets_ce,size_average=False)
    print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
    fl_loss.backward()
    ce_loss.backward()
    print(inputs_fl.grad.data)
    print(inputs_ce.grad.data)
