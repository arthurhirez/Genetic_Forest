import numpy as np
import uuid

seed = 12
np.random.seed(seed)


class Node:
  def __init__(self,
               name: str,
               value : float or int,
               cond_type : int = 0,
               depth : int = 0,
               leaf : bool = False):

    self.name = name
    self.value = value
    self.cond_type = cond_type

    if cond_type == 0: #Categorical
      self.symbol = '<='
    elif cond_type == 1: #Scalar
      self.symbol = '='

    self.depth = depth
    self.leaf = leaf

    self.left = None    #true
    self.right = None   #false

    self.identifier = uuid.uuid4()


  def decision(self,
               x: float or int):

    return x <= self.value if self.cond_type == 0 else  self.value == x
