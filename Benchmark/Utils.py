import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

seed = 12
np.random.seed(seed)


def model_pred(n_class: int,
               agents: list,
               data: np.ndarray):
    pred = []
    probs = []
    for x in data:
        prob = np.zeros(n_class)

        for tree in agents:
            result = tree.make_predict(x)
            prob[result] += 1

        pred.append(np.argmax(prob))
        probs.append(prob)

    return pred, probs


def model_prob(n_class: int,
               agents: list,
               data: np.ndarray,
               agent_weight: np.ndarray):
    predictions = []
    probability = []
    for x in data:
        prob = np.zeros(n_class)

        for tree, weight in zip(agents, agent_weight):
            result = tree.make_predict(x)
            prob[result] += weight

        predictions.append(np.argmax(prob))
        probability.append(prob)
    return predictions, probability


def score(y_pred, y):
    s = sum(y_pred == y) / len(y)

    print(f'Score :: {s}')
    return s


def plot_models(historic):
    gen_concat = []
    elite_concat = []
    for key, value in historic.items():
        mean_gen = value['gen_pop'].groupby('epoch')['score'].mean().reset_index()
        mean_gen['id'] = key
        mean_elite = value['elite_pop'].groupby('epoch')['score'].mean().reset_index()
        mean_elite['id'] = key

        gen_concat.append(mean_gen)
        elite_concat.append(mean_elite)

    genpop_concat = pd.concat(gen_concat, ignore_index = True)
    elite_concat = pd.concat(elite_concat, ignore_index = True)

    populations = {"General population": genpop_concat,
                   "Elite population": elite_concat}

    fig, axs = plt.subplots(1, 2, figsize = (20, 8))

    for i, (title, population) in enumerate(populations.items()):
        ax = axs[i]
        sns.lineplot(data = population, x = 'epoch', y = 'score', hue = 'id', palette = 'Set1', ax = ax)
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)

    return fig

def plot_params(res_AG):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    sns.barplot(data=res_AG, x='n_models', y='score_pos', hue='n_epochs', ax=axes[0][0])
    axes[0][0].set_title('Bar Plot with Hues for Different n_epochs')
    axes[0][0].set_xlabel('n_models')
    axes[0][0].set_ylabel('score_pos')
    axes[0][0].legend(title='n_epochs', loc = 'lower right')
    axes[0][0].set_ylim(0.8, 1)

    sns.barplot(data=res_AG, x='n_models', y='score_pos', hue='n_population', ax=axes[0][1])
    axes[0][1].set_title('Bar Plot with Hues for Different n_population')
    axes[0][1].set_xlabel('n_models')
    axes[0][1].set_ylabel('score_pos')
    axes[0][1].legend(title='n_population', loc = 'lower right')
    axes[0][1].set_ylim(0.8, 1)

    sns.barplot(data=res_AG, x='n_models', y='time', hue='n_epochs', ax=axes[1][0])
    axes[1][0].set_title('Bar Plot with Hues for Different n_epochs')
    axes[1][0].set_xlabel('n_models')
    axes[1][0].set_ylabel('Execution time')
    axes[1][0].legend(title='n_epochs', loc = 'upper left')

    sns.barplot(data=res_AG, x='n_models', y='time', hue='n_population', ax=axes[1][1])
    axes[1][1].set_title('Bar Plot with Hues for Different n_population')
    axes[1][1].set_xlabel('n_models')
    axes[1][1].set_ylabel('Execution time')
    axes[1][1].legend(title='n_population', loc = 'upper left')
    plt.tight_layout()
    plt.show()

    fig.savefig(f'estudo_param.png')