import numpy as np
import torch
import neural_model as nn
import dataset
import options_parser as op
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(options):
    seed = options.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    dim = options.dim
    n = options.num_samples

    net = nn.Net(dim=dim)
    d = torch.load('trained_cnn_model.pth')
    net.load_state_dict(d['state_dict'])
    net.double()

    for p in net.parameters():
        params = p.data.detach().numpy().reshape(-1)
        plt.hist(params, bins=100)
        plt.savefig('plots/regression_histogram.pdf')

    w = dataset.get_hyperplane(dim)
    x, y = dataset.sample_points(w, n=n)
    x = torch.from_numpy(x.transpose())
    #print(w)
    #print(net(x))
    #print(y)

if __name__ == "__main__":
    options = op.setup_options()
    main(options)
