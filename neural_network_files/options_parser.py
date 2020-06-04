import argparse


def setup_options():
    options = argparse.ArgumentParser()
    options.add_argument('-r', action='store', dest='seed',
                         default=10, type=int)
    options.add_argument('-d', action='store', dest='dim',
                             default=20, type=int)
    options.add_argument('-n', action='store', dest='num_samples',
                             default=10, type=int)
    return options.parse_args()
