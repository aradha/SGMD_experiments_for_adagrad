import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import random

# 20 , dim 2 , SEED 10 works

num_per_class = 200
dims = 1000
history = np.zeros((1, dims))
SEED = 17
num_timesteps = 100000

def step(w_t, X, y, lr, L, loss_at_zero):
    global history
    grad = ((y - w_t @ X) @ X.T)
    history += np.power(grad, 2)
    conditioner = 1 / np.power(history, .5)
    w_t = w_t + lr * conditioner * grad
    al = np.sqrt(np.min(history))
    au = np.sqrt(np.max(history))

    loss = 1/num_per_class * .5 * np.sum(np.power(y - w_t @ X, 2))
    #print("Loss: ", loss)

    #print(u, L, al, au, next_lr)
    return w_t, al, au


def norm(x):
    return np.sqrt(np.sum(np.power(x, 2)))


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    w = np.random.uniform(size=(1, dims))
    X = np.random.uniform(size=(dims, num_per_class))
    y = w @ X

    if dims > num_per_class:
        l, v = eig(X.T @ X)
        l = sorted(l.real, reverse=True)
    else:
        l, v = eig(X @ X.T)
        l = sorted(l.real, reverse=True)

    u_estimate = l[-1] / num_per_class
    print("U: ", u_estimate )

    sol = np.zeros((1, dims))
    loss_at_zero = 1/num_per_class * .5 * np.sum(np.power(y - sol @ X, 2))

    #grad = (y - sol @ X) @ X.T

    sup_L = 0
    for xi in X.T:
        norm_xi = np.sqrt(np.sum(np.power(xi, 2)))
        sup_L = max(norm_xi, sup_L)
    #print(sup_L)

    losses = [loss_at_zero]

    for t in range(num_timesteps):
        xidx = random.choice(list(range(len(X.T))))
        xi = X.T[xidx].reshape(-1, 1)
        yi = y.T[xidx].reshape(1, 1)
        if t % 100000 == 0:
            print("Epoch: ", t)
        if t == 0:
            grad = ((yi - sol @ xi) @ xi.T)
            al = np.sqrt(np.min(np.power(grad, 2)))
            au = np.sqrt(np.max(np.power(grad, 2)))
            print("Suggested ADAGRAD LR: ",
                  .8 * al**2 * u_estimate /  (sup_L**2 * au))
            lr = .8 * al**2 * u_estimate / (sup_L**2 * au)

        sol, al, au = step(sol, xi, yi, lr, sup_L, loss_at_zero)
        lr = .8 * al**2 * u_estimate / (sup_L**2 * au)
        loss = 1/num_per_class * .5 * np.sum(np.power(y - sol @ X, 2))
        losses.append(loss)
    print("Final LR: ", lr)
    print(loss)
    losses = [np.log(loss) for loss in losses]
    plt.plot(list(range(len(losses))), losses, '-g',
             label='Our Step Size (Adaptive)')

    """
    sol = np.zeros((1, dims))
    loss_at_zero = 1/num_per_class * .5 * np.sum(np.power(y - sol @ X, 2))

    losses = [loss_at_zero]

    for t in range(num_timesteps):
        xidx = random.choice(list(range(len(X.T))))
        xi = X.T[xidx].reshape(-1, 1)
        yi = y.T[xidx].reshape(1, 1)

        if t == 0:
            grad = ((yi - sol @ xi) @ xi.T)

            al = np.sqrt(np.min(np.power(grad, 2)))
            au = np.sqrt(np.max(np.power(grad, 2)))
            print("Suggested ADAGRAD LR: ",
                  .8 * al**2 * u_estimate /  (sup_L**2 * au))
            lr = .8 * al**2 * u_estimate / (sup_L**2 * au)

        sol, al, au = step(sol, xi, yi, lr, sup_L, loss_at_zero)
        #lr = .8 * al**2 * u_estimate / (sup_L**2 * au)
        loss = 1/num_per_class * .5 * np.sum(np.power(y - sol @ X, 2))
        losses.append(loss)
    print("Final LR: ", lr)
    print(loss)
    losses = [np.log(loss) for loss in losses]
    plt.plot(list(range(len(losses))), losses, '-b',
             label='Our Step Size (Fixed)')
    #"""


    sol = np.zeros((1, dims))
    loss_at_zero = 1/num_per_class * .5 * np.sum(np.power(y - sol @ X, 2))

    losses = [loss_at_zero]

    lr = .1

    for t in range(num_timesteps):
        if t % 100000 == 0:
            print("Epoch: ", t)
        xidx = random.choice(list(range(len(X.T))))
        xi = X.T[xidx].reshape(-1, 1)
        yi = y.T[xidx].reshape(1, 1)
        sol, al, au = step(sol, xi, yi, lr, sup_L, loss_at_zero)
        loss = 1/num_per_class * .5 * np.sum(np.power(y - sol @ X, 2))
        losses.append(loss)

    print(loss)
    losses = [np.log(loss) for loss in losses]
    plt.plot(list(range(len(losses))), losses, '-r',
             label='Step Size of 0.1')
    plt.legend()
    plt.savefig('training_loss.pdf')


if __name__ == "__main__":
    main()
