import numpy as np
import pandas as pd
from tree import Tree
from AG import Genetic

from tqdm import tqdm


class Model:
    def __init__(self,
                 feature_names: list,
                 target_names: list,
                 n_models: int = 50,
                 max_depth: int = 16,
                 pop_size: int = 150,
                 epochs: int = 50):

        self.mean = None
        self.std = None
        self.species = []


        for i in range(n_models):
            model = Tree(feature_names, target_names, max_depth).create()
            specie = Genetic(pop_size, epochs, model)
            self.species.append(specie)

    def fit(self,
            X_train: np.array,
            y_train: np.array):

        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)

        # Normalize train data
        X_train = self.norm(X_train)

        for specie in tqdm(self.species):
            specie.train(X_train, y_train)

    def predict(self, X: np.array):
        # Normalize new data with train estimators
        X = self.norm(X)

        # Compute mean prediction between the best individuals on each specie
        y_predict = np.mean([specie.model.predict(X, one_hot=True) for specie in self.species], axis=0)

        return np.argmax(y_predict, axis=1)

    def norm(self, X_data: np.array):
        return (X_data - self.mean) / self.std
