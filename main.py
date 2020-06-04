import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# 20 , dim 2 , SEED 10 works
# 20 dims, 2000 ex
# 1000 dims, 200 ex
num_per_class = 200
dims = 1000
history = np.zeros((1, dims))
SEED = 10


def step(w_t, X, y, lr, L, loss_at_zero):
    global history
    grad = ((y - w_t @ X) @ X.T)

    history += np.power(grad, 2)
    conditioner = 1. / np.power(history, .5)
    w_t = w_t + lr * conditioner * grad
    al = np.sqrt(np.min(history))
    au = np.sqrt(np.max(history))
    loss = .5 * np.sum(np.power(y - w_t @ X, 2))

    return w_t, al


def norm(x):
    return np.sqrt(np.sum(np.power(x, 2)))


def main():
    np.random.seed(SEED)
    w = np.random.uniform(size=(1, dims))
    X = np.random.uniform(size=(dims, num_per_class))
    y = w @ X

    l, _ = eig(X @ X.T)
    l = sorted(l, reverse=True)

    num_timesteps = 3000

    #sol = np.random.uniform(size=(1,dims))
    sol = np.zeros((1, dims))
    loss_at_zero = .5 * np.sum(np.power(y - sol @ X, 2))

    grad = (y - sol @ X) @ X.T
    L = l[0].real
    al = np.sqrt(np.min(np.power(grad, 2)))

    print("Suggested ADAGRAD LR: ", 2 * al /  L)
    lr = (2 * al / L * .99)

    adaptive_losses = [loss_at_zero]


    for t in range(num_timesteps):

        sol, al = step(sol, X, y, lr, L, loss_at_zero)
        lr =  2 * al / L *.99
        loss = .5 * np.sum(np.power(y - sol @ X, 2))
        adaptive_losses.append(loss)
    print("Final Learning Rate: ", lr)
    adaptive_losses = [np.log(loss) for loss in adaptive_losses]
    plt.plot(list(range(len(adaptive_losses))), adaptive_losses, '-g',
             label='Our Step Size (Adaptive)')
    #plt.savefig('training_loss.png')
    print(loss)

    sol = np.zeros((1, dims))
    loss_at_zero = .5 * np.sum(np.power(y - sol @ X, 2))
    grad = (y - sol @ X) @ X.T
    L = l[0].real
    al = np.sqrt(np.min(np.power(grad, 2)))
    print("Suggested ADAGRAD LR: ", 2 * al /  L)
    lr = (2 * al / L * .99)

    fixed_losses = [loss_at_zero]

    for t in range(num_timesteps):

        sol, al = step(sol, X, y, lr, L, loss_at_zero)
        loss = .5 * np.sum(np.power(y - sol @ X, 2))
        fixed_losses.append(loss)
    fixed_losses = [np.log(loss) for loss in fixed_losses]
    plt.plot(list(range(len(fixed_losses))), fixed_losses, '-b',
             label='Our Step Size (Fixed)')


    sol = np.zeros((1, dims))
    loss_at_zero = .5 * np.sum(np.power(y - sol @ X, 2))

    lr = .1

    default_losses = [loss_at_zero]

    for t in range(num_timesteps):

        sol, al = step(sol, X, y, lr, L, loss_at_zero)

        loss = .5 * np.sum(np.power(y - sol @ X, 2))
        default_losses.append(loss)
    default_losses = [np.log(loss) for loss in default_losses]
    plt.plot(list(range(len(default_losses))), default_losses, '-r',
             label='Step Size of 0.1')
    plt.legend()
    plt.savefig('training_loss.pdf')


if __name__ == "__main__":
    main()
