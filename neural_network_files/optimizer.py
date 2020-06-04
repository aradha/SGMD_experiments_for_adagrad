import torch
from torch.optim.optimizer import Optimizer, required


class MirrorDescent(Optimizer):

    def __init__(self, params, lr=required, q=2):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.q = q
        defaults = dict(lr=lr)
        super(MirrorDescent, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MirrorDescent, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                psi = torch.pow(torch.abs(p.data), self.q-1) * torch.sign(p.data)
                #np.power(np.abs(self.W1), q-1) * np.sign(self.W1)
                #np.abs(np.power(np.abs(arg1), 1/(q-1))) * np.sign(arg1)
                arg = psi -group['lr'] * d_p
                #p.data.add_(-group['lr'], d_p)
                #p.data.add(1, psi)
                p.data = torch.pow(torch.abs(arg), 1/(self.q-1)) * torch.sign(arg)
        return loss
