import numpy as np
from vq import VectorQuantization
import matplotlib.pyplot as plt

def square_euclid(x, y):
    return np.sum((x - y) ** 2, axis=-1)

class NeuralGas(VectorQuantization):
    def __init__(self, n_prototypes=10, eta0=0.3, lambda0=5.0, 
                 eta_min=0.05, lambda_min=0.5, n_epochs=10):
        super().__init__(n_prototypes, eta=eta0, n_epochs=n_epochs)
        self.eta0 = eta0
        self.lambda0 = lambda0
        self.eta_min = eta_min
        self.lambda_min = lambda_min

    def fit(self, X):
        self.init_prototypes(X)
        n_samples = X.shape[0]
        T = self.n_epochs * n_samples
        t = 0
        self.errors = []

        for epoch in range(self.n_epochs):
            for x in np.random.permutation(X):
                eta_t = self.eta0 * (self.eta_min / self.eta0) ** (t / T)
                lambda_t = self.lambda0 * (self.lambda_min / self.lambda0) ** (t / T)

                distances = square_euclid(x, self.prototypes)
                ranking = np.argsort(distances)

                for rank, i in enumerate(ranking):
                    h = np.exp(-rank / lambda_t)
                    self.prototypes[i] += eta_t * h * (x - self.prototypes[i])

                t += 1

            error = self.score(X)
            self.errors.append(error)
            print(f"Epoka {epoch + 1}/{self.n_epochs}, błąd: {error:.6f}")

        return self

    def init_prototypes(self, X):
        self.prototypes = np.random.permutation(X)[:self.k].copy()
        return self

    def find_nearest_prototype(self, x):
        dist = square_euclid(x, self.prototypes)
        return np.argmin(dist)

    def score(self, X):
        return np.mean([
            np.min(np.sum((x - self.prototypes) ** 2, axis=1))
            for x in X
        ])
