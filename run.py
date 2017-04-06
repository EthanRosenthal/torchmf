import numpy as np
import scipy.sparse as sp
import torch

from torchmf import BaseModule, BasePipeline
import utils


if __name__ == '__main__':

    df = utils.read_movielens_df()

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    interactions = np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1]-1, row[2]-1] = row[3]

    train, test = utils.train_test_split(interactions)
    train = sp.coo_matrix(train)
    test = sp.coo_matrix(test)

    pipeline = BasePipeline(train, test_data=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=20,
                            verbose=True)
    pipeline.fit()
