import numpy as np

from tqdm.notebook import tqdm, trange
from tree import Tree, make_predict
from genetic import Genetic
from sklearn.utils import resample


class GenForest:
    """
    A Genetic Forest (GenForest) class for training and predicting using an ensemble of decision trees with genetic algorithms.

    Parameters:
    - features (list): List of feature names.
    - targets (numpy.ndarray): Array of target values.
    - n_species (int): Number of species (ensemble size).
    - n_features (int): Number of features.
    - n_agents (int): Number of agents in each species.
    - epochs (int): Number of epochs for training genetic algorithms.
    - n_deaths (int): Number of agents to be replaced in each epoch.
    - rounds_deaths (int): Number of rounds for agent replacements.
    - seed (int): Random seed for reproducibility.

    Attributes:
    - features (list): List of feature names.
    - targets (numpy.ndarray): Array of target values.
    - n_targets (int): Number of target classes.
    - species (list): List to store individual species (decision trees).
    - n_species (int): Number of species (ensemble size).
    - species_fitness (list): List to store fitness values of each species.
    - species_best_fitness (numpy.ndarray): Array of best fitness values among species.
    - n_features (int): Number of features.
    - n_agents (int): Number of agents in each species.
    - epochs (int): Number of epochs for training genetic algorithms.
    - n_bests (int): Number of best species to keep.
    - bests_idx (numpy.ndarray): Indices of the best species.
    - bests_species (numpy.ndarray): Array of the best species.
    - seed (int): Random seed for reproducibility.
    - n_deaths (int): Number of agents to be replaced in each epoch.
    - rounds_deaths (int): Number of rounds for agent replacements.

    Methods:
    - __init__(): Initializes the GenForest with the given parameters and creates the ensemble of decision trees.
    - fit(x_train, y_train): Fits the GenForest model on the training data using genetic algorithms.
    - predict(data, bests=False): Makes predictions on the input data using ensemble of decision trees.
    - accuracy(y_predict, y): Computes the accuracy of predicted values compared to true labels.

    Note: This class relies on external functions/classes such as 'Tree', 'Genetic', and 'make_predict'.
    """
    def __init__(self, features: list, targets: np.ndarray,
                 n_species: int, n_features: int,
                 n_agents: int, epochs: int,
                 n_deaths: int, rounds_deaths: int,
                 seed: int = 123):
        """
        Initializes the GenForest with the given parameters and creates the ensemble of decision trees.

        Args:
        - features (list): List of feature names.
        - targets (numpy.ndarray): Array of target values.
        - n_species (int): Number of species (ensemble size).
        - n_features (int): Number of features.
        - n_agents (int): Number of agents in each species.
        - epochs (int): Number of epochs for training genetic algorithms.
        - n_deaths (int): Number of agents to be replaced in each epoch.
        - rounds_deaths (int): Number of rounds for agent replacements.
        - seed (int): Random seed for reproducibility.
        """

        # Initialize parameters
        self.features = features
        self.targets = targets
        self.n_targets = self.targets.shape[0]

        self.species = []
        self.n_species = n_species
        self.species_fitness = []
        self.species_best_fitness = None

        self.n_features = n_features
        self.n_agents = n_agents
        self.epochs = epochs

        # Calculate number of best species to keep
        aux = int(self.n_species * 0.1)
        self.n_bests = aux if aux % 2 != 0 else aux + 1
        self.bests_idx = np.arange(0, self.n_species)
        self.bests_species = None
        self.seed = seed

        # Set random seed and create the ensemble of decision trees
        np.random.seed(self.seed)
        for _ in range(self.n_species):
            specie = Tree(self.features, self.targets, self.n_features)
            specie.create()
            self.species.append(specie.matrix)

        self.n_deaths = n_deaths
        self.rounds_deaths = rounds_deaths

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the GenForest model on the training data using genetic algorithms.

        Args:
        - x_train (numpy.ndarray): Training features.
        - y_train (numpy.ndarray): Training target values.
        """
        # Resample training data for each epoch
        x_samples = []
        y_samples = []
        train = np.append(x_train, np.expand_dims(y_train, axis=1), axis=1)

        for _ in trange(self.epochs, desc='Resample'):
            sample = resample(train, n_samples=x_train.shape[0], replace=True, stratify=train, random_state=self.seed)
            y_samples.append(sample[:, -1])
            x_samples.append(sample[:, :-1])

        x_train = np.array(x_samples)
        y_train = np.array(y_samples)

        # Train each species using genetic algorithms
        np.random.seed(self.seed)
        for specie in tqdm(self.species, desc='Train Model :: '):
            genetic = Genetic(self.epochs, self.n_agents,
                              specie, n_deaths=self.n_deaths,
                              rounds_deaths=self.rounds_deaths)
            genetic.train(y_train, x_train, self.seed)
            self.species_fitness.append(genetic.fitness)

        # Store the best fitness values among species
        self.species_best_fitness = np.array([cell[-1] for cell in self.species_fitness])

        # Select the best species
        self.bests_idx = np.argsort(self.species_best_fitness)[-self.n_bests:]
        self.bests_species = np.array(self.species)[self.bests_idx]

    def predict(self, data: np.ndarray, bests: bool = False):
        """
        Makes predictions on the input data using ensemble of decision trees.

        Args:
        - data (numpy.ndarray): Input data for prediction.
        - bests (bool): If True, use best species for voting. If False, use soft voting.

        Returns:
        - predict (list): Predicted labels.
        """
        predict = []
        # Best voting
        if bests:
            for cell in data:
                probs = np.zeros(self.n_targets)

                for specie in self.bests_species:
                    result = make_predict(specie, cell)
                    probs[result] += 1

                predict.append(np.argmax(probs))

        # Soft voting
        else:
            # Compute weights based on species fitness
            error = np.power(1 - self.species_best_fitness, 2)
            weights = 1 / (error + 1e-9)
            weights = np.array(weights)
            weights = (weights - np.mean(weights)) / np.std(weights)
            for cell in data:
                probs = np.zeros(self.n_targets)

                for idx, specie in enumerate(self.species):
                    result = make_predict(specie, cell)
                    probs[result] += weights[idx]

                predict.append(np.argmax(probs))

        return predict


def accuracy(y_predict: list, y: list):
    """
    Computes the accuracy of predicted values compared to true labels.

    Args:
    - y_predict (list): Predicted labels.
    - y (list): True labels.

    Returns:
    - accuracy (float): Accuracy value.
    """
    return sum(y_predict == y) / len(y)
