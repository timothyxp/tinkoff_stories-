import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pipeline.logging.logger import logger
from scipy.sparse import coo_matrix


class ALS:
    def __init__(self, n_factors=32, n_ter=100, learning_rate=0.1):
        self.n_factors = n_factors
        self.n_iter = n_ter
        self.lr = learning_rate

        self.user_mapper = None
        self.item_mapper = None

        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix):
        logger.info(f"have matrix with shape {user_item_matrix.shape}")

        if self.user_factors is None and self.item_factors is None:
            self.user_factors = nn.Parameter(torch.randn(user_item_matrix.shape[0], self.n_factors) / self.n_factors)
            self.item_factors = nn.Parameter(torch.randn(user_item_matrix.shape[1], self.n_factors) / self.n_factors)

        values = user_item_matrix.data
        indices = np.vstack((user_item_matrix.row, user_item_matrix.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = user_item_matrix.shape

        user_item_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        criterion = nn.MSELoss()
        optimizer = Adam([self.user_factors, self.item_factors], lr=self.lr, amsgrad=True)
        start_time = time.time()

        for iter_num in range(self.n_iter):
            self.user_factors.requires_grad = False
            self.item_factors.requires_grad = False

            if iter_num % 2 == 0:
                self.user_factors.requires_grad = True
            else:
                self.item_factors.requires_grad = True

            predicted_matrix = torch.matmul(self.user_factors, self.item_factors.permute(1, 0))

            loss = criterion(predicted_matrix, user_item_matrix)
            logger.debug("Iter {}/{}, Loss {}, Time {}".format(
                iter_num, self.n_iter, loss.cpu().data.numpy(), time.time() - start_time))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        logger.info("Optimization finished, Loss {}, Time {}".format(
            loss.cpu().data.numpy(), time.time() - start_time))

    def predict(self, customer_id, story_id):
        u_index = self.user_mapper.get(customer_id, None)
        i_index = self.item_mapper.get(story_id, None)

        if u_index is None or i_index is None:
            return None

        return float(torch.dot(self.user_factors[u_index], self.item_factors[i_index]))


    def __repr__(self):
        return self.__class__.__name__