
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Node import Node
from Tree import Tree



class Genetic_Ensemble:

    def __init__(self,
                 agents: ['tree'],
                 epochs: int,
                 pop_size,
                 x_data,
                 y_data,
                 bootstrap,
                 mutation_rate: float = 4e-2):

        # Parameters AG
        self.epochs = epochs
        self.mutation_rate = mutation_rate
        self.save_rate = mutation_rate
        self.agents = agents
        self.n_agents = len(self.agents)

        self.x = x_data
        self.y = y_data
        self.bootstrap = bootstrap

        # Properties AG
        self.population_size = pop_size
        self.population = self.start_pop()
        self.pop_init = self.population.copy()
        self.scores = []
        self.no_improvement = 0
        self.patience_genocide = 0

        # Best individual
        self.best = self.population[0]
        self.best_score = 0
        self.best_dict = {}

        self.results = pd.DataFrame(columns = ['epoch', 'agent', 'mutation', 'best_score', 'score'])

        self.pred = {}
        self.prob = {}

        self.population_history = []
        self.best_history = []

    def start_pop(self):
        random_matrix = np.random.randn(self.population_size, self.n_agents)
        for row in random_matrix:
            while np.random.rand() < 0.40:
                idx = np.random.randint(0, self.n_agents)
                row[idx] = 0

        normalized_matrix = random_matrix / random_matrix.sum(axis = 1, keepdims = True)

        return normalized_matrix

    def pred_prob(self,
                  data,
                  n_class: int):
        predictions = {}
        probability = {}

        for i, individual in enumerate(self.population):
            predictions[i] = []
            probability[i] = []
            for x in data:
                prob = np.zeros(n_class)
                for tree, weight in zip(self.agents, individual):
                    result = tree.make_predict(x)
                    prob[result] += weight

                predictions[i].append(np.argmax(prob))
                probability[i].append(prob)

        self.pred = predictions
        self.prob = probability
        return predictions, probability

    def set_best(self, agent, score, idx):
        self.best = agent.copy()  # salva o melhor para cross over
        if (self.best_score == score):
            self.best_dict[score] = agent.copy()
        self.best_score = score
        self.mutation_rate = self.save_rate
        self.no_improvement = 0

    def evaluation_function(self, y_pred, y):
        return sum(y_pred == y) / len(y)

    def evaluation(self,
                   epoch: int):
        if self.bootstrap:
            x_ts_prob, y_ts_prob = resample(self.x, self.y, stratify = self.y)
        else:
            x_ts_prob, y_ts_prob = self.x, self.y

        y_pred_prob, _ = self.pred_prob(data = x_ts_prob, n_class = len(np.unique(y_ts_prob)))
        self.scores = []
        for i, individual in enumerate(self.population):

            score = self.evaluation_function(y_pred_prob[i], y_ts_prob)
            self.results.loc[len(self.results)] = [epoch, i, self.mutation_rate, self.best_score, score]
            self.scores.append(score)
            if (score > self.best_score):
                self.set_best(individual, score, i)
            # if (score < self.worst_score): self.set_worst(agent, score, i)

    def predation_rand(self, index = None):
        if index == None:  # so mata o pior
            self.population[self.worst_idx] = np.random.normal(size = (1, self.n_nodes))
        else:  # mata um batch de ruins
            new = np.random.randn(1, self.n_agents)
            self.population[index] = new / np.sum(new)

    def cross_and_mutation(self):
        for i, agent in enumerate(self.population):
            # if general_pop:

            # elitismo para população geral -> repordução sexuada
            cross_best = (agent + self.best) / 2
            idx = np.random.randint(0, self.n_agents)
            mutation = np.random.normal() * self.mutation_rate
            cross_best[
                idx] += 2 * mutation if np.random.rand() < 0.50 else -2 * mutation  # espaço de busca = 2 (+- valor eixo)
            self.population[i] = (cross_best / np.sum(cross_best)).copy()

    def train(self):

        flag_pred = 0
        for i in range(self.epochs):

            # se a media do desvio padrao de todos os genes da população menor que 0.05 -> baixa diversidade
            low_diversity = (np.mean(self.calculate_diversity(self.population)) < 0.05)
            flag_pred += 1 if low_diversity else 0

            # Criterio de aumento da mutacao caso fit esteja estagnado
            self.no_improvement += 1
            self.patience_genocide += 1

            # A cada 20 epoch faz grupo "elite" -> buscar ajuste fino
            # if ((i % 30 == 0) or (i == 10)):
            #   self.set_elite()
            #
            self.evaluation(epoch = i)  # Avalia os indivíduos

            if self.no_improvement == 7:
                self.mutation_rate *= 1.15
                self.no_improvement = 0

            if self.patience_genocide == 40:
                self.population = self.start_pop()
                self.mutation_rate = self.save_rate
                self.patience_genocide = 0
                self.no_improvement = 0
                flag_pred = 0

            if flag_pred == 10:
                # substituindo um batch de ruins
                batch_worst_size = int(self.population_size / 7)  # 14% da populacao
                batch_worst = sorted(range(len(self.scores)), key = lambda x: self.scores[x])[:batch_worst_size]
                for bad_indiv in batch_worst:
                    self.predation_rand(bad_indiv)
                flag_pred = 0

            self.cross_and_mutation()
            # self.results_train.loc[len(self.results)] = [i, self.best_score, self.mutation_rate]

            self.population_history.append(self.population.copy())
            self.best_history.append(self.best_score.copy())

    def calculate_diversity(self, population):
        return np.std(population, axis = 0).mean()

    def track_diversity(self):
        diversities = [self.calculate_diversity(pop) for pop in self.population_history]
        return diversities

    def plot_diversity(self):
        fig, ax = plt.subplots(1, 1, figsize = (10, 6))
        diversities = self.track_diversity()
        ax.hlines(y = 0.1, xmin = 0, xmax = self.epochs)
        ax.plot(range(len(diversities)), diversities, marker = 'o', linestyle = '-')
        ax.set_title('Diversity Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Diversity')
        ax.grid(True)
        return fig, ax

    def track_convergence_speed(self):
        convergence_speed = [self.best_history[i + 1] - self.best_history[i] for i in range(len(self.best_history) - 1)]
        return convergence_speed

    def plot_convergence_speed(self):
        fig, ax = plt.subplots(1, 1, figsize = (10, 6))
        convergence_speed = self.track_convergence_speed()
        ax.plot(range(len(convergence_speed)), convergence_speed, marker = 'o', linestyle = '-')
        ax.set_title('Convergence Speed')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Improvement in Best Score')
        return fig, ax

    def plot_diagnosis(self, gen_pop = False):
        fig, axs = plt.subplots(1, 2, figsize = (20, 6))
        ax = axs[0]
        diversities = self.track_diversity()
        ax.hlines(y = 0.1, xmin = 0, xmax = self.epochs)
        ax.plot(range(len(diversities)), diversities, marker = 'o', linestyle = '-')
        ax.set_title('Diversity Over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Diversity')
        ax.grid(True)

        ax = axs[1]
        convergence_speed = self.track_convergence_speed()
        ax.plot(range(len(convergence_speed)), convergence_speed, marker = 'o', linestyle = '-')
        ax.set_title('Convergence Speed')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Improvement in Best Score')

    def plot_pop(self):
        population = self.results

        fig, ax = plt.subplots(1, 1, figsize = (20, 8))

        for agent in population['agent'].unique():
            data = population.loc[population['agent'] == agent]
            ax.plot(data['epoch'], data['score'], alpha = 0.33)

        mean_scores = population.groupby('epoch')['score'].mean().reset_index()
        max_scores = population.groupby('epoch')['score'].max().reset_index()

        ax.plot(mean_scores['epoch'], mean_scores['score'], color = 'black', lw = 4, label = 'Mean Score')
        ax.plot(max_scores['epoch'], max_scores['score'], color = 'red', lw = 3, ls = 'dashed', label = 'Mean Score')
        ax.plot(max_scores['epoch'], self.best_history, color = 'blue', lw = 4, ls = ':', label = 'Best Score')
        ax.set_title("General population")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)