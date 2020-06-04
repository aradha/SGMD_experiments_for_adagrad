import trainer
import numpy as np
import torch
import dataset
import options_parser as op

def main(options):
    seed = options.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    d = options.dim
    n = options.num_samples

    x, y = dataset.make_dataset(d, n)

    trainer.train_net(x, y)

if __name__ == "__main__":
    options = op.setup_options()
    main(options)
