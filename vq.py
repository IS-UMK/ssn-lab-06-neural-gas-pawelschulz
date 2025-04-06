import numpy as np
from sklearn.base import BaseEstimator

def square_euclid(x, y):

    return np.sum((x - y)**2, axis=1)

class VectorQuantization(BaseEstimator):

    def __init__(self, n_prototypes=10, eta=0.1, n_epochs=10):
        self.k=n_prototypes
        self.eta = eta
        self.n_epochs = n_epochs

    def init_prototypes(self, X):

        # ustawia początkową pozycję prototypów
        
        return self

    def find_nearest_prototype(self, x):
        
        # zwraca indeks najbliższego prototypu do x 
        
        return

    def fit(self, X):

        self.init_prototypes(X)
        self.errors = [ self.score(X)]

        # algorytm uczenia wektorowej kwantyzacji  
    
        return self

    def predict(self, X):
        # zwraca wektor indeksów przypisujących wektory z X do prototypów
        return 

    def score(self, X):
        # zwraca miarę dokładności kwantyzacji, średnią odległość od zwycięskich prototypów
        
        return 