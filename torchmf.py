import collections

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

import numpy as np
import scipy.sparse as sp


class BaseModule(nn.Module):
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0):
        self.n_users = n_users
        self.n_items = n_items
        super(BaseModule, self).__init__()
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
        self.dropout_p = dropout_p
        assert self.dropout_p >= 0, 'Dropout cannot be negative'
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = self._pass
            
    @staticmethod
    def _pass(arg):
        return arg
        
    def forward(self, users, items):
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)
        preds = (self.dropout(ues) * self.dropout(uis)).sum(1)

        preds += self.user_biases(users)
        preds += self.item_biases(items)
        return preds
    
    def __call__(self, *args):
        return self.forward(*args)


class BPRModule(BaseModule):
    
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0):
        super(BPRModule, self).__init__(
            n_users,
            n_items,
            n_factors=n_factors,
            dropout_p=dropout_p
        )
        
    def forward(self, users, pos_items, neg_items):
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(pos_items) - self.item_embeddings(neg_items)
        preds = (self.dropout(ues) * self.dropout(uis)).sum(1)

        preds += self.user_biases(users)
        preds += self.item_biases(pos_items) - self.item_biases(neg_items)
        return preds


class BaseMF:

    def __init__(self,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 loss_function=nn.MSELoss(size_average=False),
                 n_epochs=10,
                 random_seed=None,
                 batch_size=32,
                 early_stopping=False,
                 stopping_window=3
                 ):
        self._optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer(self.model.parameters(), **self._optimizer_kwargs)
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        if random_seed is not None:
            self._random_seed = random_seed
            np.random.seed(self._random_seed)
        self.batch_size = batch_size
        self.warm_start = False
        self.last_epoch = 0
        self.loss = collections.defaultdict(list)
        self.early_stopping = early_stopping
        self.stopping_window = stopping_window
  
    def print_row(self):
        row = 'epoch: {} | train: {}'.format(self.loss['epoch'][-1], self.loss['train'][-1])
        if 'test' in self.loss:
            row += ' | test: {}'.format(self.loss['test'][-1])
        print(row)
        
    def stop_early(self):
        if self.last_epoch == 0:
            self._min_loss = self.loss['test'][0]
        if self.loss['test'][-1] < self._min_loss:
            self._min_loss = self.loss['test'][-1]
        if self.last_epoch <= self.stopping_window:
            return False
        return self._min_loss < np.min(self.loss['test'][-self.stopping_window:])
    
    def fit(self, X_train, y_train=None, X_test=None, y_test=None):
        indices = np.arange(X_train.nnz)
        np.random.shuffle(indices)
        rows = X_train.row.astype(np.long)
        cols = X_train.col.astype(np.long)
        vals = X_train.data.astype(np.float32)
        for epoch in range(self.last_epoch, self.last_epoch + self.n_epochs):
            train_loss = self._fit_epoch(indices, rows, cols, vals)
            self.loss['train'].append(train_loss)
            if X_test is not None:
                test_loss = self.validation_loss(X_test, y_test)
                self.loss['test'].append(test_loss)
            self.loss['epoch'].append(epoch)
            self.last_epoch = epoch
            self.print_row()
            if self.early_stopping:
                if self.stop_early():
                    break

    def _fit_batch(self, rows, cols, vals, indices):
        pass

    def _fit_epoch(self, indices, rows, cols, vals):
        pass

    def validation_loss(self, X_test, y_test=None):
        pass

    def predict(self, users, items):
        return self.model.forward(users, items)
        

class ExplicitMF(BaseMF):
    
    def __init__(self,
                 model_args,
                 model_kwargs={},
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 loss_function=nn.MSELoss(size_average=False),
                 n_epochs=10,
                 random_seed=None,
                 batch_size=32,
                 early_stopping=False,
                 stopping_window=3
                 ):
        self._model_kwargs = model_kwargs
        self.model = BaseModule(*model_args, **self._model_kwargs)
        super(ExplicitMF, self).__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            loss_function=loss_function,
            n_epochs=n_epochs,
            random_seed=random_seed,
            batch_size=batch_size,
            early_stopping=early_stopping,
            stopping_window=stopping_window
        )
        
    def _fit_batch(self, rows, cols, vals, indices):
        r = Variable(torch.from_numpy(rows[indices]))
        c = Variable(torch.from_numpy(cols[indices]))
        v = Variable(torch.from_numpy(vals[indices]))

        self.optimizer.zero_grad()
        preds = self.model(r, c)

        loss = self.loss_function(preds, v)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]
        
    def _fit_epoch(self, indices, rows, cols, vals):
        total_loss = torch.Tensor([0])
        train_losses = []
        test_losses = []

        for i in range(0, len(indices), self.batch_size):
            idxs = indices[i:i + self.batch_size]
            loss = self._fit_batch(rows, cols, vals, idxs)
            total_loss += loss
        if i < len(indices) - 1:
            # Leftover
            idxs = indices[i:]
            loss = self._fit_batch(rows, cols, vals, idxs)
            total_loss += loss

        total_loss /= len(indices)
        return total_loss[0]
    
    def validation_loss(self, X_test, y_test=None):
        rows = X_test.row.astype(np.long)
        cols = X_test.col.astype(np.long)
        vals = X_test.data.astype(np.float32)

        r = Variable(torch.from_numpy(rows))
        c = Variable(torch.from_numpy(cols))
        v = Variable(torch.from_numpy(vals))

        setattr(self.model.dropout, 'train', False)

        preds = self.model(r, c)
        loss = self.loss_function(preds, v)

        setattr(self.model.dropout, 'train', True)
        return loss.data[0] / len(r)


class BPRMF(BaseMF):
    
    def __init__(self,
                 model_args,
                 model_kwargs={},
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 loss_function=nn.MSELoss(size_average=False),
                 n_epochs=10,
                 random_seed=None,
                 batch_size=32,
                 early_stopping=False,
                 stopping_window=3
                 ):
        self._model_kwargs = model_kwargs
        self.model = BPRModule(*model_args, **self._model_kwargs)
        super(BPRMF, self).__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            loss_function=loss_function,
            n_epochs=n_epochs,
            random_seed=random_seed,
            batch_size=batch_size,
            early_stopping=early_stopping,
            stopping_window=stopping_window
        )

    def sample_triplet(self, batch_size):
        return np.random.choice(self.model.n_items, batch_size, replace=True)
        
    def _fit_batch(self, rows, cols, vals, indices):
        r = Variable(torch.from_numpy(rows[indices]))
        pos = Variable(torch.from_numpy(cols[indices]))
        neg = Variable(torch.from_numpy(self.sample_triplet(len(indices))))
        v = Variable(torch.from_numpy(vals[indices]))

        self.optimizer.zero_grad()
        preds = self.model(r, pos, neg)

        loss = self.loss_function(preds, v)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]
        
    def _fit_epoch(self, indices, rows, cols, vals):
        total_loss = torch.Tensor([0])
        train_losses = []
        test_losses = []

        for i in range(0, len(indices), self.batch_size):
            idxs = indices[i:i + self.batch_size]
            loss = self._fit_batch(rows, cols, vals, idxs)
            total_loss += loss
        if i < len(indices) - 1:
            # Leftover
            idxs = indices[i:]
            loss = self._fit_batch(rows, cols, vals, idxs)
            total_loss += loss

        total_loss /= len(indices)
        return total_loss[0]
    
    def validation_loss(self, X_test, y_test=None):
        rows = X_test.row.astype(np.long)
        cols = X_test.col.astype(np.long)
        vals = X_test.data.astype(np.float32)

        r = Variable(torch.from_numpy(rows))
        pos = Variable(torch.from_numpy(cols))
        neg = Variable(torch.from_numpy(self.sample_triplet(len(rows))))
        v = Variable(torch.from_numpy(vals))

        preds = self.model(r, pos, neg)
        loss = self.loss_function(preds, v)
        return loss.data[0] / len(r)

    def loss_function(self, preds, v):
        return (1 - nn.Sigmoid(preds)).sum()
