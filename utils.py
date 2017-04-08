import os
import requests
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

"""
Shamelessly stolen from
https://github.com/maciejkula/triplet_recommendations_keras
"""


def train_test_split(interactions, n=10):
    """
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.

    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    test = np.zeros(interactions.shape)
    train = interactions.copy()
    for user in range(interactions.shape[0]):
        test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                             size=n,
                                             replace=False)
        train[user, test_interactions] = 0.
        test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test


def _get_data_path():
    """
    Get path to the movielens dataset file.
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data')
    if not os.path.exists(data_path):
        print('Making data path')
        os.mkdir(data_path)
    return data_path


def _download_movielens(dest_path):
    """
    Download the dataset.
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    print('Downloading MovieLens data')

    with open(os.path.join(dest_path, 'ml-100k.zip'), 'wb') as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)

    with zipfile.ZipFile(os.path.join(dest_path, 'ml-100k.zip'), 'r') as z:
        z.extractall(dest_path)


def read_movielens_df():
    path = _get_data_path()
    zipfile = os.path.join(path, 'ml-100k.zip')
    if not os.path.isfile(zipfile):
        _download_movielens(path)
    fname = os.path.join(path, 'ml-100k', 'u.data')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(fname, sep='\t', names=names)
    return df


def get_movielens_interactions():
    df = read_movielens_df()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    interactions = np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions


def get_movielens_train_test_split(implicit=False):
    interactions = get_movielens_interactions()
    if implicit:
        interactions = (interactions >= 4).astype(np.float32)
    train, test = train_test_split(interactions)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)
    return train, test
