import numpy as np
from tqdm import tqdm

from tree import Tree


class Genetic:

    def __init__(self,
                 pop_size: int,
                 epochs: int,
                 model: Tree,
                 mutation_rate: float = 4e-2):

        self.epochs = epochs
        self.pop_size = pop_size
        self.population = []

        self.model = model
        self.n_nodes = len(model.nodes)
        self.best = None
        self.best_score = None
        self.fitness = []

        self.mutation_rate = mutation_rate
        self.save_rate = mutation_rate
        self.no_improvement = 0

    def start_pop(self):
        self.population = np.random.normal(size=(self.pop_size, self.n_nodes))

        self.best = self.population[0]
        self.best_score = 0

    def evaluation_function(self,
                            data: np.ndarray,
                            target: np.ndarray,
                            sample_size: int):
        y_predict = self.model.predict(data)
        return (y_predict == target).sum() / sample_size

    def evaluation(self,
                   data: np.ndarray,
                   target: np.ndarray):

        sample_size = len(target)
        for agent in self.population:

            for node, value in zip(self.model.nodes, agent):
                node.value = value

            score = self.evaluation_function(data, target, sample_size)

            if score > self.best_score:
                self.best = agent.copy()
                self.best_score = score

                self.mutation_rate = self.save_rate
                self.no_improvement = 0

    def cross_and_mutation(self):

        for agent in self.population:

            # cross better and agent
            agent = (agent + self.best) / 2

            # mutation
            idx = np.random.randint(0, self.n_nodes)
            if np.random.rand() <= .5:
                agent[idx] += np.random.normal() * self.mutation_rate
            else:
                agent[idx] -= np.random.normal() * self.mutation_rate

    def train(self,
              data: np.ndarray,
              target: np.ndarray,
              bar_train: bool = False,
              desc: str = 'Train'):

        self.start_pop()

        epochs = range(self.epochs)

        if bar_train:
            epochs = tqdm(epochs, desc=desc)

        for _ in epochs:

            self.evaluation(data, target)
            self.no_improvement += 1
            if self.no_improvement == 5:
                self.mutation_rate += self.save_rate / 5
                self.no_improvement = 0

            self.cross_and_mutation()

        for node, value in zip(self.model.nodes, self.best):
            node.value = value
