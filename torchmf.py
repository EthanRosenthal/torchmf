import collections
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.autograd
from torch.autograd import Variable
from torch import nn
import torch.utils.data as data
from tqdm import tqdm


class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, train_data, test_data=None, train=True):
        self.train = train
        self.train_data = train_data.tocoo()
        self.test_data = test_data.tocoo()
        self.n_users = self.train_data.shape[0]
        self.n_items = self.train_data.shape[1]

        self.train_row = torch.from_numpy(self.train_data.row.astype(np.long))
        self.train_col = torch.from_numpy(self.train_data.col.astype(np.long))
        self.train_val = torch.from_numpy(self.train_data.data.astype(np.float32))

        self.test_row = torch.from_numpy(self.test_data.row.astype(np.long))
        self.test_col = torch.from_numpy(self.test_data.col.astype(np.long))
        self.test_val = torch.from_numpy(self.test_data.data.astype(np.float32))

    def __getitem__(self, index):
        if self.train:
            row = self.train_row[index]
            col = self.train_col[index]
            val = self.train_val[index]
        else:
            row = self.test_row[index]
            col = self.test_col[index]
            val = self.test_val[index]

        return (row, col), val

    def __len__(self):
        if self.train:
            return self.train_data.nnz
        else:
            return self.test_data.nnz


class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, train_data, test_data=None, train=True):
        self.train = train
        self.train_data = train_data.tocoo()
        self.test_data = test_data.tocoo()
        self.n_users = self.train_data.shape[0]
        self.n_items = self.train_data.shape[1]

        self.train_row = torch.from_numpy(self.train_data.row.astype(np.long))
        self.train_col = torch.from_numpy(self.train_data.col.astype(np.long))
        self.train_val = torch.from_numpy(self.train_data.data.astype(np.float32))

        train_csr = train_data.tocsr()
        if not train_csr.has_sorted_indices:
            train_csr.sort_indices()
        self.train_indptr = train_csr.indptr
        self.train_indices = train_csr.indices

        self.test_row = torch.from_numpy(self.test_data.row.astype(np.long))
        self.test_col = torch.from_numpy(self.test_data.col.astype(np.long))
        self.test_val = torch.from_numpy(self.test_data.data.astype(np.float32))

        test_csr = test_data.tocsr()
        if not test_csr.has_sorted_indices:
            test_csr.sort_indices()
        self.test_indptr = test_csr.indptr
        self.test_indices = test_csr.indices

    def __getitem__(self, index):
        if self.train:
            row = self.train_row[index]
            found = False
            while not found:
                neg_col = np.random.randint(self.n_items)
                if self.not_rated(row, neg_col, self.train_indptr,
                                  self.train_indices):
                    found = True

            pos_col = self.train_col[index]
            val = self.train_val[index]
        else:
            row = self.test_row[index]
            found = False
            while not found:
                neg_col = np.random.randint(self.n_items)
                if self.not_rated(row, neg_col, self.test_indptr,
                                  self.test_indices):
                    found = True

            pos_col = self.test_col[index]
            val = self.test_val[index]

        return (row, pos_col, neg_col), val

    def __len__(self):
        if self.train:
            return self.train_data.nnz
        else:
            return self.test_data.nnz

    @staticmethod
    def not_rated(row, col, indptr, indices):
        # similar to use of bsearch in lightfm
        start = indptr[row]
        end = indptr[row + 1]
        searched = np.searchsorted(indices[start:end], col, 'right')
        if searched >= (end - start):
            # After the array
            return False
        return col != indices[searched]  # Not found


class BaseModule(nn.Module):
    """
    Base module for explicit matrix factorization.
    """
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 loss_function=nn.MSELoss(size_average=False),
                 sparse=False):
        """

        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        loss_function
            Torch loss function. Not technically needed here, but it's nice
            to attach for later usage.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(BaseModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)
        
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.loss_function = loss_function
        
    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:

        user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices

        Returns
        -------
        preds : np.ndarray
            Predicted ratings.

        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(1)

        return preds
    
    def __call__(self, *args):
        return self.forward(*args)


def bpr_loss(preds, vals):
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()


class BPRModule(BaseModule):
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 margin=1.0,
                 loss_function=bpr_loss,
                 sparse=False):
        super(BPRModule, self).__init__(
            n_users,
            n_items,
            n_factors=n_factors,
            dropout_p=dropout_p,
            loss_function=loss_function,
            sparse=sparse
        )
        self.margin = margin

    def forward(self, users, pos_items, neg_items):
        ues = self.user_embeddings(users)
        uis = (self.item_embeddings(pos_items)
               - self.item_embeddings(neg_items))
        preds = (self.dropout(ues) * self.dropout(uis)).sum(1)

        preds += self.user_biases(users)
        preds += self.item_biases(pos_items)
        preds -= self.item_biases(neg_items)
        return preds

    def predict(self, users, items):
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)
        preds = (self.dropout(ues) * self.dropout(uis)).sum(1)

        preds += self.user_biases(users)
        preds += self.item_biases(items)
        return preds


class BasePipeline:
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train_data,
                 test_data=None,
                 model=BaseModule,
                 n_factors=40,
                 batch_size=32,
                 dropout_p=0.02,
                 sparse=False,
                 lr=0.01,
                 weight_decay=0.,
                 optimizer=torch.optim.Adam,
                 loss_function=nn.MSELoss(size_average=False),
                 n_epochs=10,
                 verbose=False,
                 random_seed=None,
                 interaction_class=Interactions,
                 hogwild=False,
                 num_workers=0):
        self.train_interactions = interaction_class(train_data,
                                               test_data=test_data,
                                               train=True)
        self.num_workers = num_workers
        self.train_loader = data.DataLoader(
            self.train_interactions, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers
        )
        if test_data is not None:
            self.test_interactions = interaction_class(train_data,
                                                  test_data=test_data,
                                                  train=False)
            self.test_loader = data.DataLoader(
                self.test_interactions, batch_size=batch_size, shuffle=True,
            )

        self.n_users = train_data.shape[0]
        self.n_items = train_data.shape[1]
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        if sparse:
            assert weight_decay == 0.0
        self.model = model(self.n_users,
                           self.n_items,
                           n_factors=self.n_factors,
                           dropout_p=self.dropout_p,
                           loss_function=self.loss_function,
                           sparse=sparse)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        self.warm_start = False
        self.losses = collections.defaultdict(list)
        self.verbose = verbose
        self.hogwild = hogwild
        if random_seed is not None:
            if self.hogwild:
                random_seed += os.getpid()
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def break_grads(self):
        for param in self.model.parameters():
            # Break gradient sharing
            if param.grad is not None:
                param.grad.data = param.grad.data.clone()

    def fit(self):
        if self.hogwild:
            self.break_grads()
        for epoch in range(1, self.n_epochs + 1):
            self.losses['train'].append(self._fit_epoch(epoch))
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.test_interactions is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)

    def _fit_epoch(self, epoch=1):
        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, col), val) in pbar:
            row = Variable(row)
            col = Variable(col)
            val = Variable(val).float()
            self.optimizer.zero_grad()
            preds = self.model(row, col)
            loss = self.model.loss_function(preds, val)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]
            batch_loss = loss.data[0] / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= len(self.train_interactions)
        return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = Variable(row)
            col = Variable(col)
            val = Variable(val).float()
            preds = self.model(row, col)
            loss = self.model.loss_function(preds, val)
            total_loss += loss.data[0]
        total_loss /= len(self.test_interactions)
        return total_loss[0]


class BPRPipeline(BasePipeline):
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train_data,
                 model=BPRModule,
                 loss_function=bpr_loss,
                 interaction_class=PairwiseInteractions,
                 **kwargs):
        super(BPRPipeline, self).__init__(
            train_data, model=model, loss_function=loss_function,
            interaction_class=interaction_class, **kwargs
        )

    def _fit_epoch(self, epoch=1):
        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, pos_col, neg_col), val) in pbar:
            row = Variable(row)
            pos_col = Variable(pos_col)
            neg_col = Variable(neg_col)
            val = Variable(val).float()
            self.optimizer.zero_grad()
            preds = self.model(row, pos_col, neg_col)
            loss = self.model.loss_function(preds, val)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]
            batch_loss = loss.data[0] / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= len(self.train_interactions)
        return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        return self.auc()

    def auc(self):
        self.model.eval()
        aucs = []
        items_init = torch.from_numpy(np.arange(self.n_items, dtype=np.long))
        users_init = torch.ones(self.n_items).long()
        for row in range(self.n_users):
            users = Variable(users_init * np.long(row))
            items = Variable(items_init)
            preds = self.model.predict(users, items)

            start = self.test_loader.dataset.test_indptr[row]
            end = self.test_loader.dataset.test_indptr[row + 1]
            actuals = self.test_loader.dataset.test_indices[start:end]

            if len(actuals) == 0:
                continue
            y_test = np.zeros(self.n_items)
            y_test[actuals] = 1
            aucs.append(roc_auc_score(y_test, preds.data.numpy()))
        return np.sum(aucs) / len(aucs)

