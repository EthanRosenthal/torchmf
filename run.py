import argparse
import pickle

import torch

from torchmf import BaseModule, BasePipeline, BPRPipeline
import utils


def explicit():
    train, test = utils.get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=40,
                            verbose=True, random_seed=2017)
    pipeline.fit()


def implicit():
    # train, test = utils.get_movielens_train_test_split(implicit=True)

    train, test, x, y, z = pickle.load(open('cache.p', 'rb'))

    pipeline = BPRPipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()


def hogwild():
    # train, test = utils.get_movielens_train_test_split(implicit=True)
    train, test, x, y, z = pickle.load(open('cache.p', 'rb'))

    pipeline = BPRPipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017, hogwild=True,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()


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

