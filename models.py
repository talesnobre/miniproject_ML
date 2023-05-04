import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
import numpy.linalg as LA
import pandas as pd
import random
from random import sample 
from tqdm import tqdm

class PocketPLA:
    def __init__(self, iterations=1000, n_min=50, n_max=200):
        self.iterations = iterations
        self.n_min = n_min
        self.n_max = n_max

    def __str__(self):
        return "Pocket PLA"

    def fit(self, X, y, iterations=None):
        if iterations is not None:
            self.iterations = iterations

        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(X.shape[1])
        best_error = len(y)
        best_w = self.w

        for j in tqdm(range(self.iterations)):
            n = random.randint(self.n_min, self.n_max)
            indexes = np.random.randint(len(X) - 1, size=n)
            X_ = X[indexes]
            y_ = y[indexes]

            for i in range(n):
                if np.sign(np.dot(self.w, X_[i])) != y_[i]:
                    self.w = self.w + X_[i] * y_[i]
                    e_in = self.error_in(X_, y_)
                    if best_error > e_in:
                        best_error = e_in
                        best_w = self.w
                        if best_error == 0:
                            break

        self.w = best_w

    def get_w(self):
        return self.w

    def set_w(self, w):
        self.w = w

    def h(self, x):
        return np.sign(np.dot(self.w, x))

    def error_in(self, X, y):
        return np.mean(np.sign(np.dot(self.w, X.T)) != y)

    def predict(self, X):
        return [self.h(x) for x in X]

    
class LinearRegression():

    def __init__(self, iter=None):
        self.iter = iter

    def __str__(self):
        return "Linear Regression"

    def fit(self, X, y):
        h = np.dot(X.T, X)
        g = np.dot(X.T, y)
        self.w = np.dot((inv(h)), g)

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T))
    
    def get_w(self):
        return self.w

class LogisticRegression:
    
    def __init__(self, eta=0.1, iter=1000, batch_size=2048):
        self.eta = eta
        self.iter = iter
        self.batch_size = batch_size
        self.w = None

    def __str__(self):
        return "Logistic Regression"
    
    def fit(self, X, y, iter=None, lamb=1e-6):

        if iter is not None:
            self.iter = iter

        N, d = X.shape
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        w = np.zeros(d)

        for t in tqdm(range(self.iter)):
            if self.batch_size < N:
                rand_indexes = np.random.choice(N, self.batch_size, replace=False)
                X_batch, y_batch = X[rand_indexes], y[rand_indexes]
                N_batch = self.batch_size
            else:
                X_batch, y_batch = X, y
                N_batch = N

            sigm = 1 / (1 + np.exp(y_batch * np.dot(w, X_batch.T).reshape(-1, 1)))
            gt = - 1 / N_batch * np.sum(X_batch * y_batch * sigm, axis=0) 

            if np.linalg.norm(gt) < 1e-10:
                break
            
            w -= self.eta  * gt 

        self.w = w
    
    def predict_prob(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    def predict(self, X):
        pred = self.predict_prob(X)
        y = np.where(pred >= 0.5, 1, -1)
        return y 

    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w


class OneVsAll:

    def __init__(self, model=None, digits=None, iters=None):
        self.model = model 
        self.digits = digits
        self.all_w = []
        self.iters = iters

    def fit(self, X_train, y_train):
        for i, d in enumerate(self.digits[:-1]):
            if i == 0:
                y_train_i = np.where(y_train == d, 1, -1)
            else:
                X_train = np.delete(X_train, np.where(y_train == d_anterior), axis=0)
                y_train = np.delete(y_train, np.where(y_train == d_anterior))
                y_train_i = np.where(y_train == d, 1, -1)

            if self.iters is None:
                self.model.fit(X_train, y_train_i)
            else:
                self.model.fit(X_train, y_train_i, iter=self.iters[i])

            self.all_w.append(self.model.get_w())
            d_anterior = d
        
    def predict(self, X):
        predictions = []
        for x in X:
            predicted_digits = []
            for i, d in enumerate(self.digits[:-1]):
                if np.sign(self.all_w[i] @ x) == 1:
                    predicted_digits.append(d)
                    break
            if len(predicted_digits) == 0:
                predicted_digits.append(self.digits[-1])
            predictions.append(predicted_digits)
        return np.array(predictions)

    def get_all_w(self):
        return self.all_w

    def set_all_w(self, all_w):
        self.all_w = all_w