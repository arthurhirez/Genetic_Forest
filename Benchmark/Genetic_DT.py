from sklearn.utils import resample

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Node import Node
from Tree import Tree


class Genetic:

    def __init__(self,
                 pop_size: int,
                 epochs: int,
                 model: Tree,
                 elitism: bool,
                 predation: bool,
                 genocide: bool,
                 x_data,
                 y_data,
                 bootstrap,
                 mutation_rate: float = 4e-2,
                 documentation = False):

        # Parameters AG
        self.epochs = epochs
        self.pop_size = pop_size
        self.model = model
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.genocide = genocide
        self.predation = predation
        self.save_rate = mutation_rate
        self.bootstrap = bootstrap

        # Properties AG
        self.n_nodes = len(model.nodes)
        self.population = []
        self.scores = []
        self.no_improvement = 0
        self.patience_genocide = 0
        self.x = x_data
        self.y = y_data

        self.elite_pop = []
        self.elite_mutation = .05

        # Best individual
        self.best = None
        self.best_score = None
        self.elite_is_best = []
        self.population_is_best = []

        # Worst individual
        self.worst = None
        self.worst_score = None
        self.worst_idx = 0

        # Save data
        self.documentation = documentation

        self.tournam_history = []
        self.population_history = []
        self.best_history = []
        self.results = pd.DataFrame(columns = ['epoch', 'agent', 'mutation', 'best_score', 'score'])
        self.results_elite = pd.DataFrame(columns = ['epoch', 'agent', 'best_score', 'score', 'individuo'])
        # self.results_train = pd.DataFrame(columns = ['epoch', 'best_score', 'mutation_rate'])
        self.control_cross = []

    def start_pop(self):
        self.population = np.random.normal(size = (self.pop_size, self.n_nodes))
        self.elite_pop = np.random.normal(size = (self.n_nodes, self.n_nodes))

        self.best = self.population[0]
        self.best_score = 0

        self.worst = self.population[0]
        self.worst_score = 1
        self.worst_idx = 0

    def set_elite(self):
        for i in range(self.n_nodes):
            cross_best = self.best
            cross_best[i] *= (1 + 5 * self.elite_mutation) if np.random.rand() < 0.50 else (1 - 5 * self.elite_mutation)
            self.elite_pop[i] = cross_best.copy()

    def set_best(self, agent, score, idx):
        self.best = agent.copy()  # salva o melhor para cross over
        self.best_score = score
        self.mutation_rate = self.save_rate
        self.no_improvement = 0

    def set_worst(self, agent, score, idx):
        self.worst = agent
        self.worst_score = score
        self.worst_idx = idx

    def evaluation_function(self,
                            data: np.ndarray,
                            target: np.ndarray,
                            sample_size: int):

        return sum(self.model.predict(data) == target) / sample_size

    def evaluation(self,
                   data: np.ndarray,
                   target: np.ndarray,
                   epoch: int):

        sample_size = len(target)
        self.scores = []

        # Avaliacao elite
        for i, agent in enumerate(self.elite_pop):
            for node, value in zip(self.model.nodes, agent):
                node.value = value
            score = self.evaluation_function(data, target, sample_size)
            self.results_elite.loc[len(self.results_elite)] = [epoch, i, self.best_score, score, agent.copy()]
            if (score > self.best_score):
                self.set_best(agent, score, i)
                self.elite_is_best.append(epoch)

        # Avaliacao populacao geral
        for i, agent in enumerate(self.population):
            for node, value in zip(self.model.nodes, agent):
                node.value = value
            score = self.evaluation_function(data, target, sample_size)
            self.results.loc[len(self.results)] = [epoch, i, self.mutation_rate, self.best_score, score]
            self.scores.append(score)
            if (score > self.best_score):
                self.set_best(agent, score, i)
                self.population_is_best.append(epoch)
            if (score < self.worst_score): self.set_worst(agent, score, i)

    def tournament(self, general = True):
        idx_father = np.random.randint(0, self.pop_size)
        if general:
            idx_mother = np.random.randint(0, self.pop_size)
            if idx_father == idx_mother: idx_father = np.random.randint(0, self.pop_size)
        else:
            batch_best_size = int(self.pop_size / 7)  # 14% da populacao
            batch_indexes = np.argsort(self.scores)[::-1]
            batch_best = batch_indexes[:batch_best_size]
            idx_mother = batch_best[np.random.randint(0, batch_best_size)]
            if idx_father == idx_mother: idx_father = np.random.randint(0, self.pop_size)

        son = (self.population[idx_father] + self.population[idx_mother]) / 2
        return son

    def cross_and_mutation(self, general_pop = True):
        population = self.population if general_pop else self.elite_pop

        # elitismo para população geral -> repordução sexuada
        if general_pop:
            new_pop = []
            for i in range(self.pop_size):
                if (np.random.rand() < 0.30) or self.elitism:  # elitismo -> melhor de todos
                    cross_best = (population[i] + self.best) / 2
                    self.control_cross.append('best')
                else:
                    cross_best = self.tournament(general = True)
                    self.control_cross.append('tourn')
                idx = np.random.randint(0, self.n_nodes)
                mutation = np.random.normal() * self.mutation_rate
                cross_best[
                    idx] += 2 * mutation if np.random.rand() < 0.50 else -2 * mutation  # espaço de busca = 2 (+- valor eixo)
                new_pop.append(cross_best.copy())
            new_pop = np.vstack(new_pop)

            for i in range(self.pop_size):
                population[i] = new_pop[i]

        # reprodução assexuada para grupo elite
        else:
            for i, agent in enumerate(population):
                for _ in range(3):  # colocar proporcional ao tamanho do cromossomo?
                    idx = np.random.randint(0, self.n_nodes)
                    agent[idx] *= (1 + self.elite_mutation) if np.random.rand() < 0.50 else (
                                1 - self.elite_mutation)  # ajuste fino -> % no valor e não no espaço de busca
                population[i] = agent.copy()

    def predation_rand(self, index = None):
        if index == None:  # so mata o pior
            self.population[self.worst_idx] = np.random.normal(size = (1, self.n_nodes))
        else:  # mata um batch de ruins
            self.population[index] = np.random.normal(size = (1, self.n_nodes))

    def train(self):
        self.start_pop()
        if self.documentation: self.population_history = [self.population.copy()]

        flag_pred = 0
        for i in range(self.epochs):
            if self.bootstrap:
                x_sample, y_sample = resample(self.x, self.y, stratify = self.y)
            else:
                x_sample, y_sample = self.x, self.y

            # se a media do desvio padrao de todos os genes da população menor que 0.05 -> baixa diversidade
            low_diversity = (np.mean(self.calculate_diversity(self.population)) < 0.05)
            flag_pred += 1 if low_diversity else 0

            # Criterio de aumento da mutacao caso fit esteja estagnado
            self.no_improvement += 1
            self.patience_genocide += 1

            # A cada 20 epoch faz grupo "elite" -> buscar ajuste fino
            if ((i % 20 == 0) or (i == 10)):
                self.set_elite()

            self.evaluation(x_sample, y_sample, i)  # Avalia os indivíduos

            if self.no_improvement == 7:
                self.mutation_rate *= 1.15
                self.no_improvement = 0

            if (self.patience_genocide == 40) and self.genocide:
                self.population = np.random.normal(size = (self.pop_size, self.n_nodes))
                self.mutation_rate = self.save_rate
                self.patience_genocide = 0
                self.no_improvement = 0
                flag_pred = 0

            if (flag_pred == 10) and self.predation:
                # substituindo um batch de ruins
                batch_worst_size = int(self.pop_size / 7)  # 14% da populacao
                batch_worst = sorted(range(len(self.scores)), key = lambda x: self.scores[x])[:batch_worst_size]
                for bad_indiv in batch_worst:
                    self.predation_rand(bad_indiv)
                flag_pred = 0
                # # Substituindo o pior de todos
                # self.worst_idx = self.scores.index(min(self.scores))
                # self.predation_rand()

            self.cross_and_mutation(general_pop = True)
            self.cross_and_mutation(general_pop = False)
            # self.results_train.loc[len(self.results)] = [i, self.best_score, self.mutation_rate]

            if self.documentation:
                self.population_history.append(self.population.copy())
                self.best_history.append(self.best_score)

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
        return fig

    def plot_pop(self):
        populations = {"General population": self.results,
                       "Elite population": self.results_elite}

        fig, axs = plt.subplots(1, 2, figsize = (20, 8))

        for i, (title, population) in enumerate(populations.items()):
            ax = axs[i]

            for agent in population['agent'].unique():
                data = population.loc[population['agent'] == agent]
                ax.plot(data['epoch'], data['score'], alpha = 0.33)

            mean_scores = population.groupby('epoch')['score'].mean().reset_index()
            max_scores = population.groupby('epoch')['score'].max().reset_index()

            ax.plot(mean_scores['epoch'], mean_scores['score'], color = 'black', lw = 4, label = 'Mean Score')
            ax.plot(max_scores['epoch'], max_scores['score'], color = 'red', lw = 3, ls = 'dashed',
                    label = 'Mean Score')
            ax.plot(max_scores['epoch'], self.best_history, color = 'blue', lw = 4, ls = ':', label = 'Best Score')
            ax.set_title(title)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
        return fig