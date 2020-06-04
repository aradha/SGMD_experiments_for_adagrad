import neural_model as network
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from copy import deepcopy
import pickle

#import optimizer as o

hist_1 = None
hist_2 = None
L_est = 0

def train_net(data, labels):
    global hist_1, hist_2

    dim = data.size()[-1]
    # Use the following to instantiate a network
    net = network.Net(dim=dim)
    for idx, p in enumerate(net.parameters()):
        if idx == 0:
            hist_1 = np.zeros(p.size())
        else:
            hist_2 = np.zeros(p.size())
    #"""
    bound = 1e-1
    for idx, param in enumerate(net.parameters()):
        init = torch.Tensor(param.size()).uniform_(-bound, bound)
        param.data = init

    #"""
    net.double()
    net.cuda()
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()),
                              lr=1e-1)



    num_epochs = 10000#1000000
    #Place your data on the GPU
    inputs = Variable(data.double())
    #inputs = Variable(data.float())
    inputs = inputs.cuda()
    targets = Variable(labels.double())
    #targets = Variable(labels.float())
    targets = targets.cuda()

    best_loss = np.float('inf')
    losses =[]
    flag = True
    for i in range(num_epochs):

        train_loss, lr = train_step(net, inputs, targets, optimizer, iteration=i)
        losses.append(train_loss)
        if i % 100 == 0:
            print(i, train_loss, best_loss, "Learning Rate: ", lr)

        # Save the best model if loss is low enough
        if train_loss < best_loss and train_loss < 1e-5:
            best_loss = train_loss
            d = {}
            d['state_dict'] = net.state_dict()
            torch.save(d, 'trained_cnn_model.pth')
    pickle.dump(losses, open('train_loss.p', 'wb'))


def train_step(net, inputs, targets, optimizer, iteration=None):
    global hist_1, hist_2
    global L_est
    # Set the network to training mode
    net.train()
    # Zero out all gradients
    net.zero_grad()
    # Compute the loss (MSE in this case)
    loss = 0.
    outputs = net(inputs)

    if iteration == 0:
        print("First output mean: ", outputs[0].mean())

    loss = torch.sum(torch.pow(outputs - targets, 2)) * .5
    # Compute backprop updates
    loss.backward()

    grad_norm = 0.
    for idx, p in enumerate(net.parameters()):
        g = p.grad
        if idx == 0:
            hist_1 += np.power(g.cpu().data.numpy(), 2)
        else:
            hist_2 += np.power(g.cpu().data.numpy(), 2)
        grad_norm += torch.sum(torch.pow(g, 2)).cpu().data.numpy().item()

    al = np.sqrt(np.min(hist_1))
    al = min(al, np.sqrt(np.min(hist_2)))

    f = loss.cpu().data.numpy().item()

    L_est = (grad_norm) / (2 * f)

    lr = 2  * al / L_est * .99

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    return loss.cpu().data.numpy().item(), lr
