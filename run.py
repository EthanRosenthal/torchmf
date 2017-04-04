import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from torchmf import ExplicitMF
import utils


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

if __name__ == '__main__':

    df = utils.read_movielens_df()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]

    train, test = train_test_split(ratings)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)

    emf = ExplicitMF((n_users, n_items), 
                     model_kwargs={'n_factors': 40, 'dropout_p': 0.02},
                     optimizer=torch.optim.Adam, 
                     optimizer_kwargs={'lr': 0.01, 'weight_decay': 0.1},
                     n_epochs=100, random_seed=2017, batch_size=1024,
                     early_stopping=True)
    emf.fit(train, X_test=test)