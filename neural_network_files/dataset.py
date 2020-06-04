import trainer
import torch
import numpy as np


def get_hyperplane(dim=10):
    w = np.random.normal(size=(1, dim))
    return w


def sample_points(plane, n=3):
    dim = plane.shape[-1]
    x = np.random.uniform(size=(dim, n))
    labels = plane @ x  + np.random.uniform(size=(1, n))
    return x, labels


def make_dataset(dim, num_samples):
    w = get_hyperplane(dim=dim)
    x, y = sample_points(w, n=num_samples)

    x = torch.from_numpy(x.transpose())
    y = torch.from_numpy(y.transpose())
    print(x.size(), y.size())
    return x, y


def make_test_dataset():
    pass
