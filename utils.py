import os
import requests
import pandas as pd
import zipfile

"""
Shamelessly stolen from
https://github.com/maciejkula/triplet_recommendations_keras
"""

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
