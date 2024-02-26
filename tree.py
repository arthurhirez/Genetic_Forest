import numpy as np

from numba import njit
from node import Node


class Tree:
    """
    A Tree class representing a decision tree.

    Parameters:
    - all_features (list): List of all feature names.
    - targets (numpy.ndarray): Array of target values.
    - n_features (int): Number of features to be used in the tree.

    Attributes:
    - n_features (int): Number of features.
    - all_features (list): List of all feature names.
    - chosen_features (numpy.ndarray): Randomly selected features for the tree.
    - features (dict): Dictionary mapping feature names to their indices.
    - targets (numpy.ndarray): Array of target values.
    - root (Node): Root node of the decision tree.
    - nodes (list): List of nodes in the decision tree.
    - matrix (numpy.ndarray): Matrix representation of the decision tree.
    - score (float): Score of the tree.

    Methods:
    - __init__(): Initializes the Tree object with the given parameters.
    - insert_node(): Inserts a node into the decision tree.
    - set_outputs(): Sets the outputs (leaf nodes) of the decision tree.
    - get_nodes(): Returns a list of all nodes in the decision tree.
    - to_matrix(): Converts the decision tree into a matrix representation.
    - create(): Creates the decision tree by inserting nodes and setting outputs.
    """

    def __init__(self, all_features: list, targets: np.ndarray,
                 n_features: int = 5):
        """
        Initializes the Tree object with the given parameters.

        Args:
        - all_features (list): List of all feature names.
        - targets (numpy.ndarray): Array of target values.
        - n_features (int): Number of features to be used in the tree.
        """
        self.n_features = n_features

        self.all_features = all_features
        self.chosen_features = np.random.choice(self.all_features, n_features, replace=True)
        self.features = {feature: self.all_features.index(feature) for feature in self.all_features}

        self.targets = targets
        self.root = None
        self.nodes = None
        self.matrix = None

        self.score = 0

    def insert_node(self, node: Node, key_name: str,
                    key_value: float or int = 0, current_depth: int = 0):
        """
        Inserts a node into the decision tree.

        Args:
        - node (Node): Current node in the tree.
        - key_name (str): Feature name for the node.
        - key_value (float or int): Feature value for the node.
        - current_depth (int): Current depth in the tree.

        Returns:
        - node (Node): Updated node after insertion.
        """
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

    def set_outputs(self, node: Node, outputs_classes: np.ndarray,
                    before_label: str = '', current_depth: int = 0,
                    direction: str = 'right'):
        """
        Sets the outputs (leaf nodes) of the decision tree.

        Args:
        - node (Node): Current node in the tree.
        - outputs_classes (numpy.ndarray): Array of output classes.
        - before_label (str): Label of the node before the current node.
        - current_depth (int): Current depth in the tree.
        - direction (str): Direction of the node in relation to its parent.

        Returns:
        - node (Node): Updated node with set outputs.
        """
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

    @staticmethod
    def get_nodes(root: Node):
        """
        Returns a list of all nodes in the decision tree.

        Args:
        - root (Node): Root node of the decision tree.

        Returns:
        - nodes_list (list): List of all nodes in the decision tree.
        """
        nodes_list: list = [root]
        queue: list = [root]

        while queue:
            node = queue.pop(0)

            if node.left:
                queue.append(node.left)
                nodes_list.append(node.left)
            if node.right:
                queue.append(node.right)
                nodes_list.append(node.right)

        return nodes_list

    def to_matrix(self):
        """
        Converts the decision tree into a matrix representation.

        Returns:
        - matrix (numpy.ndarray): Matrix representation of the decision tree.
        """
        n_nodes = len(self.nodes)
        matrix = -np.ones((n_nodes, 5), dtype=np.float64)

        # 0 :: index corresponding to the array
        # 1 :: node leaf class
        # 2 :: value in node
        # 3 :: true - left
        # 4 :: false - right

        idx = 1
        for row in range(n_nodes):
            if self.nodes[row].leaf:
                matrix[row][1] = self.nodes[row].value

            else:
                matrix[row][0] = self.features[self.nodes[row].name]
                matrix[row][2] = self.nodes[row].value
                for col in range(3, 5):
                    matrix[row][col] = idx
                    idx += 1

        return matrix

    def create(self):
        """
        Creates the decision tree by inserting nodes and setting outputs.
        """
        for feature in self.chosen_features:
            self.root = self.insert_node(self.root, feature, np.random.normal())

        self.root = self.set_outputs(self.root, self.targets)
        self.nodes = self.get_nodes(self.root)
        self.matrix = self.to_matrix()


@njit
def make_predict(matrix: np.ndarray, array: np.ndarray):
    """
    Makes predictions using the matrix representation of the decision tree.

    Args:
    - matrix (numpy.ndarray): Matrix representation of the decision tree.
    - array (numpy.ndarray): Input array for prediction.

    Returns:
    - prediction (int): Predicted label.
    """
    ID = 0
    LEAF = 1
    VALUE = 2
    TRUE = 3
    FALSE = 4

    current_node = 0
    while True:
        if matrix[current_node][ID] == -1:
            return int(matrix[current_node][LEAF])

        elif array[int(matrix[current_node][ID])] <= matrix[current_node][VALUE]:
            current_node = int(matrix[current_node][TRUE])

        else:
            current_node = int(matrix[current_node][FALSE])


@njit
def predict(matrix: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Makes predictions for a dataset using the matrix representation of the decision tree.

    Args:
    - matrix (numpy.ndarray): Matrix representation of the decision tree.
    - data (numpy.ndarray): Input data for prediction.

    Returns:
    - y_predict (numpy.ndarray): Predicted labels.
    """
    y_predict = np.zeros(data.shape[0])
    idx = 0

    for array in data:
        y_predict[idx] = make_predict(matrix, array)
        idx += 1

    return y_predict
