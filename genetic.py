import numpy as np

from numba import njit
from tree import predict


class Genetic:
    """
    Genetic class for evolving decision tree models using genetic algorithms.

    Parameters:
    - epochs (int): Number of epochs for the genetic algorithm.
    - n_agents (int): Number of agents (individuals) in the population.
    - model (numpy.ndarray): Initial decision tree model represented as a matrix.
    - mutation_rate (float): Probability of mutation in the genetic algorithm.
    - n_deaths (int): Number of agents to be replaced in each predation round.
    - rounds_deaths (int): Number of epochs between predation rounds.

    Attributes:
    - epochs (int): Number of epochs for the genetic algorithm.
    - n_agents (int): Number of agents (individuals) in the population.
    - agents (numpy.ndarray): Population of decision tree models.
    - model (numpy.ndarray): Current decision tree model being evolved.
    - n_nodes (int): Number of nodes in the decision tree model.
    - indexes (numpy.ndarray): Indices of non-leaf nodes in the decision tree model.
    - mutation_rate (float): Probability of mutation in the genetic algorithm.
    - init_mutation_rate (float): Initial mutation rate.
    - n_deaths (int): Number of agents to be replaced in each predation round.
    - rounds_deaths (int): Number of epochs between predation rounds.
    - fitness (numpy.ndarray): Array to store the fitness scores over epochs.

    Methods:
    - __init__(): Initializes the Genetic object with the given parameters.
    - _train(): Internal method for training the genetic algorithm.
    - train(): Trains the genetic algorithm to evolve the decision tree model.
    """
    def __init__(self, epochs: int, n_agents: int,
                 model: np.ndarray, mutation_rate: float = 4e-2,
                 n_deaths: int = 10, rounds_deaths: int = 20):
        """
        Initializes the Genetic object with the given parameters.

        Args:
        - epochs (int): Number of epochs for the genetic algorithm.
        - n_agents (int): Number of agents (individuals) in the population.
        - model (numpy.ndarray): Initial decision tree model represented as a matrix.
        - mutation_rate (float): Probability of mutation in the genetic algorithm.
        - n_deaths (int): Number of agents to be replaced in each predation round.
        - rounds_deaths (int): Number of epochs between predation rounds.
        """
        self.epochs = epochs
        self.n_agents = n_agents
        self.agents = None

        self.model = model
        self.n_nodes = model.shape[0]

        aux_indexes = np.array([i if model[:, :1][i] != -1 else -1 for i in range(self.n_nodes)])
        self.indexes = np.delete(aux_indexes, np.where(aux_indexes == -1))

        self.mutation_rate = mutation_rate
        self.init_mutation_rate = mutation_rate

        self.n_deaths = n_deaths
        self.rounds_deaths = rounds_deaths

        self.fitness = np.zeros(self.epochs, dtype=np.float64)

    @staticmethod
    @njit
    def _train(epochs: int, model: np.ndarray,
               fitness: np.ndarray, indexes: np.ndarray,
               mutation_rate: float, n_agents: int,
               n_nodes: int, n_deaths: int,
               rounds_deaths: int, y_test: np.ndarray,
               x_test: np.ndarray, seed: int = 123):
        """
        Internal method for training the genetic algorithm.

        Args:
        - epochs (int): Number of epochs for the genetic algorithm.
        - model (numpy.ndarray): Initial decision tree model represented as a matrix.
        - fitness (numpy.ndarray): Array to store the fitness scores over epochs.
        - indexes (numpy.ndarray): Indices of non-leaf nodes in the decision tree model.
        - mutation_rate (float): Probability of mutation in the genetic algorithm.
        - n_agents (int): Number of agents (individuals) in the population.
        - n_nodes (int): Number of nodes in the decision tree model.
        - n_deaths (int): Number of agents to be replaced in each predation round.
        - rounds_deaths (int): Number of epochs between predation rounds.
        - y_test (numpy.ndarray): True labels for evaluation.
        - x_test (numpy.ndarray): Input data for evaluation.
        - seed (int): Random seed for reproducibility.

        Returns:
        - model (numpy.ndarray): Final decision tree model evolved by the genetic algorithm.
        """
        agents = _start_agents(n_agents, n_nodes, seed)
        agents_score = np.zeros(n_agents, dtype=np.float64)

        best_agent_idx = 0
        best_agent = agents[best_agent_idx]
        best_score = 0

        start_mutation_rate = mutation_rate

        # Counter to see if there has been any improvement
        count_improvement = 0
        np.random.seed(seed)
        for idx in range(epochs):

            results = _evaluating(model, agents,
                                  agents_score, y_test[idx],
                                  x_test[idx], best_agent_idx,
                                  best_score)
            improvement = results[0]
            best_agent_idx = int(results[1])
            best_agent = agents[best_agent_idx].copy()
            best_score = results[2]
            fitness[idx] = best_score

            # Predation
            if idx % rounds_deaths == 0:
                order_idx = np.argsort(agents_score)
                worsts = order_idx[:n_deaths]
                for worst in worsts:
                    agents[worst] = np.random.normal(0, 1, (n_nodes, 1))

            _cross_and_mutation(agents, agents_score,
                                best_agent, indexes,
                                mutation_rate)

            if improvement == 0:
                count_improvement += 1
                if count_improvement == 5:
                    count_improvement = 0
                    mutation_rate += start_mutation_rate / 10
            else:
                count_improvement = 0
                mutation_rate = start_mutation_rate

        model[:, 2:3] = best_agent
        return model

    def train(self, y_test: np.ndarray, x_test: np.ndarray,
              seed: int = 123):
        """
        Trains the genetic algorithm to evolve the decision tree model.

        Args:
        - y_test (numpy.ndarray): True labels for evaluation.
        - x_test (numpy.ndarray): Input data for evaluation.
        - seed (int): Random seed for reproducibility.
        """
        self.model = self._train(self.epochs, self.model,
                                 self.fitness, self.indexes,
                                 self.mutation_rate, self.n_agents,
                                 self.n_nodes, self.n_deaths,
                                 self.rounds_deaths, y_test,
                                 x_test, seed)


@njit
def _start_agents(n_agents: int, n_nodes: int,
                  seed: int = 123) -> np.ndarray:
    """
    Initializes the agents (decision tree models) for the genetic algorithm.

    Args:
    - n_agents (int): Number of agents (individuals) in the population.
    - n_nodes (int): Number of nodes in the decision tree model.
    - seed (int): Random seed for reproducibility.

    Returns:
    - agents (numpy.ndarray): Array of initialized agents.
    """
    np.random.seed(seed)

    agents = np.zeros((n_agents, n_nodes, 1))

    for idx in range(n_agents):
        agents[idx] = np.random.normal(0, 1, (n_nodes, 1))

    return agents


@njit
def _evaluating(matrix: np.ndarray, agents: np.ndarray,
                agents_score: np.ndarray, y_test: np.ndarray,
                x_test: np.ndarray, best_agent_idx: int,
                best_score: float) -> np.ndarray:
    """
    Evaluates the fitness of agents using the provided data.

    Args:
    - matrix (numpy.ndarray): Decision tree model represented as a matrix.
    - agents (numpy.ndarray): Array of agents (decision tree models).
    - agents_score (numpy.ndarray): Array to store the fitness scores of agents.
    - y_test (numpy.ndarray): True labels for evaluation.
    - x_test (numpy.ndarray): Input data for evaluation.
    - best_agent_idx (int): Index of the best-performing agent.
    - best_score (float): Best fitness score achieved so far.

    Returns:
    - results (numpy.ndarray): Array containing information about the evaluation results.
    """
    improvement = 0
    size = len(y_test)
    for idx in range(agents.shape[0]):
        matrix[:, 2:3] = agents[idx]
        agent_predict = predict(matrix, x_test)
        score = sum(agent_predict == y_test) / size
        agents_score[idx] = score

        if score > best_score:
            best_agent_idx = idx
            best_score = score
            improvement = 1

    return np.array([improvement, best_agent_idx, best_score], dtype=np.float64)


@njit
def _cross_and_mutation(agents: np.ndarray, agents_score: np.ndarray,
                        best_agent: np.ndarray, indexes: np.ndarray,
                        mutation_rate: float):
    """
    Performs crossover and mutation operations on the population of agents.

    Args:
    - agents (numpy.ndarray): Array of agents (decision tree models).
    - agents_score (numpy.ndarray): Array containing fitness scores of agents.
    - best_agent (numpy.ndarray): Best-performing agent in the population.
    - indexes (numpy.ndarray): Indices of non-leaf nodes in the decision tree model.
    - mutation_rate (float): Probability of mutation.
    """
    aux_agents = agents.copy()
    n_agents = agents.shape[0]
    agents[0] = best_agent

    for idx in range(1, n_agents):

        # Tournament
        if np.random.rand() < .7:
            candidates = np.random.randint(0, n_agents, 4)
            father = candidates[0] if agents_score[candidates[0]] > agents_score[candidates[1]] else candidates[1]
            mother = candidates[2] if agents_score[candidates[2]] > agents_score[candidates[3]] else candidates[3]
            agents[idx] = (aux_agents[father] + aux_agents[mother]) / 2
        # Elitism
        else:
            agents[idx] = (agents[idx] + best_agent) / 2

        idx_mutation = int(np.random.choice(indexes))
        value = np.random.normal() * mutation_rate
        agents[idx][idx_mutation] += value if np.random.random() < .5 else - value
