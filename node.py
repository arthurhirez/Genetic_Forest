import uuid


class Node:
    """
    A Node class representing a node in a decision tree.

    Parameters:
    - name (str): Feature name for the node.
    - value (float or int): Feature value for the node.
    - cond_type (int): Type of condition for the node (0 for categorical, 1 for scalar).
    - depth (int): Depth of the node in the tree.
    - leaf (bool): Indicates whether the node is a leaf node.

    Attributes:
    - name (str): Feature name for the node.
    - value (float or int): Feature value for the node.
    - cond_type (int): Type of condition for the node (0 for categorical, 1 for scalar).
    - symbol (str): Symbol used for the decision condition ("<=" for categorical, "=" for scalar).
    - depth (int): Depth of the node in the tree.
    - leaf (bool): Indicates whether the node is a leaf node.
    - left (Node): Left child node.
    - right (Node): Right child node.
    - identifier (uuid.UUID): Unique identifier for the node.

    Methods:
    - decision(x): Returns the result of the decision condition for a given input value x.
    """
    def __init__(self, name: str, value: float or int,
                 cond_type: int = 0, depth: int = 0,
                 leaf: bool = False):
        """
        Initializes the Node object with the given parameters.

        Args:
        - name (str): Feature name for the node.
        - value (float or int): Feature value for the node.
        - cond_type (int): Type of condition for the node (0 for categorical, 1 for scalar).
        - depth (int): Depth of the node in the tree.
        - leaf (bool): Indicates whether the node is a leaf node.
        """

        self.name = name
        self.value = value
        self.cond_type = cond_type

        # Set the condition symbol based on cond_type
        self.symbol = '<=' if cond_type == 0 else '='

        self.depth = depth
        self.leaf = leaf

        self.left = None    # true
        self.right = None   # false

        self.identifier = uuid.uuid4()

    def decision(self, x: float or int):
        """
        Returns the result of the decision condition for a given input value x.

        Args:
        - x (float or int): Input value for the decision.

        Returns:
        - result (bool): Result of the decision condition.
        """
        return x <= self.value if self.cond_type == 0 else self.value == x
