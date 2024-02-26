# Genetic Forest Classifier

## Overview

This repository contains an implementation of a genetic algorithm for evolving decision trees for classification tasks. The decision tree structure is defined by the `Node` class, and the entire genetic algorithm is orchestrated by the `Genetic` class. 

## Classes

### 1. `Node`

The `Node` class represents a node in the decision tree. Each node has attributes such as `name`, `value`, `cond_type`, `depth`, and `leaf`. The `decision` method is used to make decisions based on the node's conditions.

### 2. `Tree`

The `Tree` class is responsible for creating the initial decision tree and defining its structure. It uses the `Node` class to represent nodes. The decision tree is then evolved using a genetic algorithm, and the best tree is determined based on classification accuracy.

### 3. `GenForest`

The `GenForest` class implements a Genetic Forest, an ensemble of decision trees using genetic algorithms. It utilizes the `Tree`, `Node`, and `Genetic` classes to train the ensemble and make predictions.

## Usage

To use this implementation, follow these steps:

```python
# Import necessary libraries and classes
import numpy as np
from gen_forest import GenForest, accuracy
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Wini data
data = load_wine()

df = pd.DataFrame(data['data'], columns=data['feature_names'])
Y = data['target']
X = df.to_numpy()

features = data['feature_names']
targets = data['target_names']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Normalizing the data
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Create a GenForest instance
model = GenForest(features, targets,
                      n_species=100, n_features=np.random.randint(6, 10),
                      n_agents=200, epochs=200,
                      n_deaths=5, rounds_deaths=20)

# Fit model
model.fit(x_train, y_train)

# Make predictions using the GenForest model
predictions = model.predict(x_test, bests=False)

# Evaluate accuracy
accuracy_value = accuracy(predictions, y_test)
print("Accuracy:", accuracy_value)
```

# How it works:
Basically, countless species of random trees are created, where we have as random:
- Tree structure
- Features of each node
- Values of each node

After the creation of the **n** species, an exclusive environment is created for each tree, that is, one species does not have contact with another. In this environment, the genetic algorithm is executed with the idea of finding the best combination of possible values for that model, with accuracy as the evaluation function. Thus, the process is repeated for each species.

![Alt text](Benchmark/Images/example1.png)

## Classification
After optimizing the models, where we will have good, average and bad species mixed together, we use the same random forest approach to classify an input, and it is possible to do it in the following ways:

#### 1) Soft voting, where the weight of each tree's vote is weighted according to the score it obtained at the time of training:
- $$error = 1 - accuracy$$
- $$weight = \frac{1}{error^2 + \epsilon}$$
- $$\text{Normalization of weights}$$

#### 2) Vote for the best, here in a simpler way using Hard Voting only for the top species that obtained the highest score in the training stage

Finally, we add up the votes and take the one with the highest probability or number of votes as the predicted class.


# Benchmark

## Context
When initially conceptualizing the experiment, we considered utilizing a Random Forest algorithm combined with the Genetic Algorithm (GA) approach. Following the initial analysis, we chose to implement a second GA to augment the results obtained from the initial algorithm.

To achieve this goal, we designed a GA responsible for ensemble learning and also selecting the most promising models trained by the first GA. This GA performs predictions using Soft Voting, which involves weighting each model to enhance the overall score.

## Initial Parameters Analysis:

Number of models: [20, 30, 35, 40, 50, 70]
Number of epochs: [75, 100, 125, 150]
Size of Population: [75, 100, 125, 150, 200]
Only a few models achieved a similar score to the Random Forest (baseline algorithm).

### Performance Analysis
| n\_models | n\_epochs | n\_population | score\_pos | time |
| :--- | :--- | :--- | :--- | :--- |
| 20 | 75 | 150 | 0.981 | 140.051 |
| 30 | 100 | 125 | 0.981 | 273.879 |
| 30 | 125 | 100 | 0.981 | 237.955 |
| 35 | 125 | 125 | 0.981 | 361.880 |
| 35 | 150 | 100 | 0.981 | 347.313 |
| 40 | 100 | 75 | 0.981 | 196.178 |
| 70 | 75 | 125 | 0.981 | 551.002 |

Considering both execution time and improvements in the score metric, we have chosen the following parameter intervals for the study:

Number of models: [30, 35]
Number of epochs: [100, 200]
Size of Population: [125, 200]

![Alt text](Benchmark/Images/estudo_param.png)

## Results

First, a study was conducted to assess the effects of the Mutation Rate, the introduction of Predation and Genocide, and the crossover method (Elitism - the best vs all and Tournament random parents).

To assess the differences between the methods, we monitored the population's Diversity (via standard deviation) and the overall improvement of the best individual across epochs.

In addition to the general population, we established a secondary group (Elite) where only mutation was applied (asexual reproduction), targeting a more refined tuning of the results.

Examples are provided below:
### Elitism w/ Predation and Genocide
![Alt text](Benchmark/Images/diagn_0.04_Elitismo_Predacao_Genocidio_1.png)
![Alt text](Benchmark/Images/pop_0.04_Elitismo_Predacao_Genocidio_1.png)

### Elitism w/o Predation and Genocide
![Alt text](Benchmark/Images/diagn_0.04_Elitismo_Sem_pred_Sem_gen_20.png)
![Alt text](Benchmark/Images/pop_0.04_Elitismo_Sem_pred_Sem_gen_20.png)

### Ensemble Genetic - Elitism w/ Predation and Genocide
![Alt text](Benchmark/Images/estudo_ag2.png)

Following this, we examined the results on two datasets while maintaining a constant Mutation rate (0.04):

Wine dataset: A simpler dataset consisting of (178x13) dimensions.
Health dataset: A more complex dataset with (2113x21) dimensions.


### Methods Analysis
|  | Taxa Mutação | Predação | Genocídio | Método\_x | best\_score\_x | Método\_y | best\_score\_y |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0.010 | Predacao | Genocidio | Elitismo | 0.972 | Torneio | 0.917 |
| 1 | 0.040 | Predacao | Genocidio | Elitismo | 0.972 | Torneio | 0.889 |
| 2 | 0.070 | Predacao | Genocidio | Elitismo | 0.833 | Torneio | 0.889 |
| 3 | 0.010 | Predacao | Sem Genocídio | Elitismo | 0.944 | Torneio | 0.944 |
| 4 | 0.040 | Predacao | Sem Genocídio | Elitismo | 0.944 | Torneio | 0.944 |
| 5 | 0.070 | Predacao | Sem Genocídio | Elitismo | 0.917 | Torneio | 0.889 |
| 6 | 0.010 | Sem Predação | Genocidio | Elitismo | 0.861 | Torneio | 0.972 |
| 7 | 0.040 | Sem Predação | Genocidio | Elitismo | 0.861 | Torneio | 0.944 |
| 8 | 0.070 | Sem Predação | Genocidio | Elitismo | 0.861 | Torneio | 0.917 |
| 9 | 0.010 | Sem Predação | Sem Genocídio | Elitismo | 1.000 | Torneio | 0.917 |
| 10 | 0.040 | Sem Predação | Sem Genocídio | Elitismo | 0.944 | Torneio | 0.944 |
| 11 | 0.070 | Sem Predação | Sem Genocídio | Elitismo | 0.944 | Torneio | 0.944 |


### Performance Analysis
|  | Taxa Mutação | Método | Predação | Genocídio | Pre\_Wine | Pos\_Wine | Pre\_Health | Pos\_Health |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0.040 | Elitismo | Predacao | Genocidio | 0.944 | 0.861 | 0.799 | 0.837 |
| 1 | 0.040 | Elitismo | Predacao | Sem Genocídio | 0.750 | 0.889 | 0.780 | 0.835 |
| 2 | 0.040 | Elitismo | Sem Predação | Genocidio | 0.861 | 0.861 | 0.780 | 0.827 |
| 3 | 0.040 | Elitismo | Sem Predação | Sem Genocídio | 0.889 | 0.833 | 0.785 | 0.827 |
| 4 | 0.040 | Torneio | Predacao | Genocidio | 0.889 | 0.889 | 0.799 | 0.827 |
| 5 | 0.040 | Torneio | Predacao | Sem Genocídio | 0.889 | 0.861 | 0.780 | 0.813 |
| 6 | 0.040 | Torneio | Sem Predação | Genocidio | 0.944 | 0.944 | 0.780 | 0.823 |
| 7 | 0.040 | Torneio | Sem Predação | Sem Genocídio | 0.889 | 0.861 | 0.790 | 0.837 |


It's evident that when the first GA performs relatively well, the second GA doesn't yield substantial improvements. However, in cases where the results of the first GA fall below expectations (e.g., less than 80%), the second GA can effectively identify more promising models and enhance the overall performance by approximately 5%.

### Video: https://youtu.be/exZEM803XCY
### Contributors
- Rafael Souza
- Arthur Hiratsuka Rezende
- João Augusto Fernandes
- Thiago Ambiel
- João Pedro Farjoun Silva
