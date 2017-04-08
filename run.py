import argparse
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.multiprocessing as mp

from torchmf import BaseModule, BasePipeline, BPRPipeline
import utils


def explicit():
    train, test = utils.get_movielens_train_test_split()
    pipeline = BasePipeline(train, test_data=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=10,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit():
    train, test = utils.get_movielens_train_test_split()
    pipeline = BPRPipeline(train, test_data=test, verbose=True,
                           batch_size=2048,
                           random_seed=2017)
    pipeline.fit()


def hogwild():
    train, test = utils.get_movielens_train_test_split()
    pipeline = BPRPipeline(train, test_data=test, verbose=True,
                           batch_size=2048, num_workers=1,
                           random_seed=2017, hogwild=True)

    num_processes = 4
    # NOTE: this is required for the ``fork`` method to work
    pipeline.model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=pipeline.fit, )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='torchmf')
    parser.add_argument('--example',
                        help='explicit, implicit, or hogwild')
    args = parser.parse_args()
    if args.example == 'explicit':
        explicit()
    elif args.example == 'implicit':
        implicit()
    elif args.example == 'hogwild':
        hogwild()
    else:
        print('example must be explicit, implicit, or hogwild')

