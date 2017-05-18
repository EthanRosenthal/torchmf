import collections
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.autograd
from torch.autograd import Variable
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data as data
from tqdm import tqdm


def flatten(l):
    if isinstance(l[0], list):
        return [y for x in l for y in x]
    return l

class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz


class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()

        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

        self.mat_csr = self.mat.tocsr()
        if not self.mat_csr.has_sorted_indices:
            self.mat_csr.sort_indices()

    def __getitem__(self, index):
        row = self.mat.row[index]
        found = False

        count = 0
        while not found:
            neg_col = np.random.randint(self.n_items)
            if self.not_rated(row, neg_col, self.mat_csr.indptr,
                              self.mat_csr.indices):
                found = True

        pos_col = self.mat.col[index]
        val = self.mat.data[index]

        return (row, pos_col, neg_col), val

    def __len__(self):
        return self.mat.nnz

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

    def get_row_indices(self, row):
        start = self.mat_csr.indptr[row]
        end = self.mat_csr.indptr[row + 1]
        return self.mat_csr.indices[start:end]


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

    def predict(self, users, items):
        return self.forward(users, items)


def bpr_loss(pos_preds, neg_preds, vals):
    sig = nn.Sigmoid()
    return (1.0 - sig(pos_preds - neg_preds)).pow(2).sum()


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
                 train,
                 test=None,
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
                 num_workers=0,
                 eval_metrics=None,
                 k=5):
        self.train = train
        self.test = test

        if hogwild:
            num_loader_workers = 0
        else:
            num_loader_workers = num_workers
        self.train_loader = data.DataLoader(
            interaction_class(train), batch_size=batch_size, shuffle=True,
            num_workers=num_loader_workers)
        if self.test is not None:
            self.test_loader = data.DataLoader(
                interaction_class(test), batch_size=batch_size, shuffle=True,
                num_workers=num_loader_workers)
        self.num_workers = num_workers
        self.n_users = self.train.shape[0]
        self.n_items = self.train.shape[1]
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

        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics
        self.k = k

    def break_grads(self):
        for param in self.model.parameters():
            # Break gradient sharing
            if param.grad is not None:
                param.grad.data = param.grad.data.clone()

    def fit(self):
        for epoch in range(1, self.n_epochs + 1):

            if self.hogwild:
                self.model.share_memory()
                processes = []
                train_losses = []
                queue = mp.Queue()
                for rank in range(self.num_workers):
                    p = mp.Process(target=self._fit_epoch,
                                   kwargs={'epoch': epoch,
                                           'queue': queue})
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                while True:
                    is_alive = False
                    for p in processes:
                        if p.is_alive():
                            is_alive = True
                            break
                    if not is_alive and queue.empty():
                        break

                    while not queue.empty():
                        train_losses.append(queue.get())
                queue.close()
                train_loss = np.mean(train_losses)
            else:
                train_loss = self._fit_epoch(epoch)

            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.test is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                for metric in self.eval_metrics:
                    func = getattr(self, metric)
                    res = func()
                    self.losses['eval-{}'.format(metric)].append(res)
                    row += 'eval-{0}: {1:^10.5f}'.format(metric, res)
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)

    def _fit_epoch(self, epoch=1, queue=None):
        if self.hogwild:
            self.break_grads()

        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, col), val) in pbar:
            self.optimizer.zero_grad()

            row = Variable(row.long())
            col = Variable(col.long())
            val = Variable(val).float()

            preds = self.model(row, col)
            loss = self.model.loss_function(preds, val)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.data[0]
            batch_loss = loss.data[0] / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train.nnz
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = Variable(row.long())
            col = Variable(col.long())
            val = Variable(val).float()

            preds = self.model(row, col)
            loss = self.model.loss_function(preds, val)
            total_loss += loss.data[0]

        total_loss /= self.test.nnz
        return total_loss[0]


class BPRPipeline(BasePipeline):
    """
    Class defining a training pipeline. Instantiates data loaders, model,
    and optimizer. Handles training for multiple epochs and keeping track of
    train and test loss.
    """

    def __init__(self,
                 train,
                 loss_function=bpr_loss,
                 interaction_class=PairwiseInteractions,
                 **kwargs):
        super(BPRPipeline, self).__init__(
            train, loss_function=loss_function,
            interaction_class=interaction_class, **kwargs
        )

    def _fit_epoch(self, epoch=1, queue=None):
        if self.hogwild:
            self.break_grads()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, pos_col, neg_col), val) in pbar:
            row = Variable(row.long())
            pos_col = Variable(pos_col.long())
            neg_col = Variable(neg_col.long())
            val = Variable(val).float()
            self.optimizer.zero_grad()

            pos_pred = self.model(row, pos_col)
            neg_pred = self.model(row, neg_col)

            loss = self.model.loss_function(pos_pred, neg_pred, val)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]
            batch_loss = loss.data[0] / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train.nnz
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])

        for batch_idx, ((row, pos_col, neg_col), val) in enumerate(self.test_loader):
            row = Variable(row.long())
            pos_col = Variable(pos_col.long())
            neg_col = Variable(neg_col.long())
            val = Variable(val).float()

            pos_pred = self.model(row, pos_col)
            neg_pred = self.model(row, neg_col)

            loss = self.model.loss_function(pos_pred, neg_pred, val)
            total_loss += loss.data[0]
        total_loss /= self.train.nnz
        return total_loss[0]

    def auc(self, train=False):
        self.model.eval()
        aucs = []
        processes = []
        mp_batch = int(np.ceil(self.n_users / self.num_workers))
        queue = mp.Queue()
        rows = np.arange(self.n_users)
        np.shuffle(rows)
        for rank in range(self.num_workers):
            start = rank * mp_batch
            end = np.min((start + mp_batch,  self.n_users))
            p = mp.Process(target=self.batch_auc,
                           args=(queue, rows[start:end]),
                           kwargs={'train': train})
            p.start()
            processes.append(p)

        while True:
            is_alive = False
            for p in processes:
                if p.is_alive():
                    is_alive = True
                    break
            if not is_alive and queue.empty():
                break

            while not queue.empty():
                aucs.append(queue.get())

        queue.close()
        for p in processes:
            p.join()
        return np.mean(aucs)

    def batch_auc(self, queue, rows, train=False):
        items_init = torch.arange(0, self.n_items).long()
        users_init = torch.ones(self.n_items).long()
        for row in rows:
            users = Variable(users_init.fill_(row))
            items = Variable(items_init)
            preds = self.model.predict(users, items)

            if train:
                actuals = self.train_loader.dataset.get_row_indices(row)
            else:
                actuals = self.test_loader.dataset.get_row_indices(row)

            if len(actuals) == 0:
                continue
            y_test = np.zeros(self.n_items)
            y_test[actuals] = 1
            queue.put(roc_auc_score(y_test, preds.data.numpy()))

    def patk(self, train=False):
        self.model.eval()
        patks = []
        processes = []
        mp_batch = int(np.ceil(self.n_users / self.num_workers))
        queue = mp.Queue()
        rows = np.arange(self.n_users)
        np.shuffle(rows)
        for rank in range(self.num_workers):
            start = rank * mp_batch
            end = np.min((start + mp_batch, self.n_users))
            p = mp.Process(target=self.batch_patk,
                           args=(queue, rows[start:end]),
                           kwargs={'train': train})
            p.start()
            processes.append(p)

        while True:
            is_alive = False
            for p in processes:
                if p.is_alive():
                    is_alive = True
                    break
            if not is_alive and queue.empty():
                break

            while not queue.empty():
                patks.append(queue.get())

        queue.close()
        for p in processes:
            p.join()
        return np.mean(patks)

    def batch_patk(self, queue, rows, train=False):
        items_init = torch.arange(0, self.n_items).long()
        users_init = torch.ones(self.n_items).long()
        for row in rows:
            users = Variable(users_init.fill_(row))
            items = Variable(items_init)

            preds = self.model.predict(users, items)
            if train:
                actuals = self.train_loader.dataset.get_row_indices(row)
            else:
                actuals = self.test_loader.dataset.get_row_indices(row)

            if len(actuals) == 0:
                continue

            top_k = np.argpartition(-np.squeeze(preds.data.numpy()), self.k)
            top_k = set(top_k[:self.k])
            true_pids = set(actuals)
            if true_pids:
                queue.put(len(top_k & true_pids) / float(self.k))
