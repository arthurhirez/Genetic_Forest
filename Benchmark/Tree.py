from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy

from Node import Node
from Utils import model_pred, model_prob, score, plot_models

seed = 12
np.random.seed(seed)

class Tree:

  def __init__(self,
               all_features: np.ndarray,
               targets: np.ndarray,
               max_depth: int = 5):

    self.all_features = all_features
    self.max_depth = max_depth

    self.chosen_features = np.random.choice(self.all_features, max_depth,
                                            replace=True)

    self.features = {feature : self.all_features.index(feature) for feature in self.chosen_features}

    self.targets = targets
    self.root = None
    self.nodes = None

    self.score = 0

  def __deepcopy__(self, obj):
        new = type(self)(copy.deepcopy(self.all_features, obj),
                         copy.deepcopy(self.chosen_features, obj),
                         copy.deepcopy(self.targets, obj),
                         copy.deepcopy(self.max_depth, obj)
                         )
        new.features = copy.deepcopy(self.features, obj)
        new.root = copy.deepcopy(self.root, obj)
        new.nodes = copy.deepcopy(self.nodes, obj)

        return new

  def get_params(self):
    return self.chosen_features, self.max_depth, self.score

  def insert_node(self,
                  node: Node,
                  key_name: str,
                  key_value: float or int = 0,
                  current_depth: int = 0):

    if node is None:
      return Node(key_name, key_value, depth=current_depth, cond_type=0)

    else:
      current_depth += 1
      direction = np.random.rand()
      if direction <= 0.5:
        node.left = self.insert_node(node.left, key_name, key_value, current_depth)
      else:
        node.right = self.insert_node(node.right, key_name, key_value, current_depth)

    return node


  def set_outputs(self,
                  node : Node,
                  outputs_classes: np.ndarray,
                  before_label: str = '',
                  current_depth : int = 0,
                  direction: str = 'right'):

    current_depth += 1

    if node is None:
      label = np.random.choice(outputs_classes)
      value = float(label.split('_')[1])
      return Node(label, int(value), depth=current_depth, leaf=True)

    else:
      if node.right and node.left is None:
        left_output_classes = np.delete(outputs_classes, np.where(outputs_classes == node.right.name))
      else:
        left_output_classes = outputs_classes
      node.left = self.set_outputs(node.left, left_output_classes, node.name, current_depth, direction='left')

      if node.left and node.right is None:
        right_output_classes = np.delete(outputs_classes, np.where(outputs_classes == node.left.name))
      else:
        right_output_classes = outputs_classes
      node.right = self.set_outputs(node.right, right_output_classes, node.name, current_depth)

    return node


  def to_nodes(self,
               node: Node,
               nodes: np.ndarray,):
    if node and not node.leaf:
      nodes.append(node)
      self.to_nodes(node.left, nodes)
      self.to_nodes(node.right, nodes)


  def get_nodes(self):
    nodes = []
    self.to_nodes(self.root, nodes)
    return nodes


  def create(self):
    for feature in self.chosen_features:
      self.root = self.insert_node(self.root, feature, np.random.normal())

    self.root = self.set_outputs(self.root, self.targets)
    self.nodes = self.get_nodes()


  def make_predict(self,
                   cell: np.ndarray):

    node = self.root
    while node:
      if node.leaf:
        return node.value

      elif node.decision(cell[self.features[node.name]]):
        node = node.left

      else:
        node = node.right


  def predict(self,
              data: np.ndarray):
    results = []
    for cell in data:
      results.append(self.make_predict(cell))

    return results


  def to_graph(self,
               graph,
               node: Node):

    if node.left is not None:
        graph.add_edge(node.identifier, node.left.identifier)
        self.to_graph(graph, node.left)

    if node.right is not None:
        graph.add_edge(node.identifier, node.right.identifier)
        self.to_graph(graph, node.right)


  def get_labels(self,
                 node: Node,
                 labels: dict):

    if node:
      labels[node.identifier] = f'{node.name} {node.symbol} {node.value:.4}' if not node.leaf else f'{node.name}'
      self.get_labels(node.left, labels)
      self.get_labels(node.right, labels)


  def show(self):
    graph = nx.Graph()
    self.to_graph(graph, self.root)

    labels = {}
    self.get_labels(self.root, labels)
    pos = graphviz_layout(graph, prog="dot")

    plt.figure(figsize=(13, 5))
    nx.draw(graph, pos,
            labels=labels,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_color="white",
            font_weight="bold",
            linewidths=0.5,
            edge_color="gray",
            style="dashed",
            bbox=dict(facecolor="black", edgecolor='black', boxstyle='round, pad=1.0'))

    plt.show()