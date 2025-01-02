import numpy as np
from typing import Literal, List, Any
from enum import Enum

Vec = List
Val = Any
Group = str

class BeliefGroups:

    def __init__(self, groups):
        self.groups = groups
    
    def add_group(self, group: Group):
        self.groups.add(group)


class KnowledgeBase:
    def __init__(self, beliefgroups: BeliefGroups, dim):
        """
        Initialize a knowledge base with a given dimensionality.
        :param dim: the dimensionality of the vectors to be stored
        """
        self.storage = dict.fromkeys(beliefgroups.groups, [])

        self.beliefgroups = beliefgroups

    def add_item(self, key: Vec, val: Val, group: Group):
        """
        Store the key-value pair in the knowledge base.
        :param key: key
        :param val: value
        """
        self.storage[group].append((key, val))

    def retrieve_given_group(
        self, key: Vec, metric: Literal['l2', 'cos', 'ip'], storage, k: int = 1) -> List[Val]:    
        """
        Retrieve the top k values from the knowledge base given a key and similarity metric.
        :param key: key
        :param metric: Similarity metric to use.
        :param k: Top k similar items to retrieve.
        :return: List of top k similar values.
        """
        distances = []
        if metric == 'l2':
            distances = [(self._sim_euclidean(key, item[0]), item[1]) for item in storage]
        elif metric == 'cos':
            distances = [(self._sim_cosine(key, item[0]), item[1]) for item in storage]
        elif metric == 'ip':
            distances = [(self._sim_inner_product(key, item[0]), item[1]) for item in storage]

        distances.sort(reverse = True)
        return [item[1] for item in distances[:k]]

    def retrieve(self, key, metric: Literal['l2', 'cos', 'ip'], group: Group, k: int = 1):
        """
        Retrieve the top k values from the knowledge base given a key and the belief group we want to search
        :param key: key
        :param metric: Similarity metric to use.
        :param group: belief group where we search for passages
        :param k: Top k similar items to retrieve.
        :return: List of top k similar values.
        """
        return self.retrieve_given_group(key, metric, self.storage[group], k)

    @staticmethod
    def _sim_euclidean(a: Vec, b: Vec) -> float:
        """
        Compute Euclidean (L2) distance between two vectors.
        :param a: Vector a
        :param b: Vector b
        :return: Similarity score
        """
        return -np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def _sim_cosine(a: Vec, b: Vec) -> float:
        """
        Compute the cosine similarity between two vectors.
        :param a: Vector a
        :param b: Vector b
        :return: Similarity score
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if((not norm_a) or (not norm_b)):
          return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def _sim_inner_product(a: Vec, b: Vec) -> float:
        """
        Compute the inner product between two vectors.
        :param a: Vector a
        :param b: Vector b
        :return: Similarity score
        """
        return np.dot(a, b)
    
def metric_exact_match(ans_pred: str, ans_true: str) -> float:
    """
    Case-sensitive answer exact match, model output is identical to the gold answer.
    :param ans_pred: Predicted answer
    :param ans_true: Ground truth answer
    :return: 1. if the answers are the same, 0. otherwise
    """
    if(ans_pred.strip() == ans_true.strip()):
      return 1.
    return 0.

def metric_f1(ans_pred: str, ans_true: str) -> float:
    """
    Case-insensitive answer F1 score. Use white-space separated words as "tokens".
    :param ans_pred: Predicted answer.
    :param ans_true: Ground truth answer.
    :return: F1 score between the predicted and ground truth answers.
    """
    pred_tokens = set(ans_pred.lower().split())
    true_tokens = set(ans_true.lower().split())

    if(len(pred_tokens) == 0 or len(true_tokens) == 0):
      return 0.
    precision = len(pred_tokens.intersection(true_tokens)) / len(pred_tokens)
    recall = len(pred_tokens.intersection(true_tokens)) / len(true_tokens)
    if precision + recall == 0:
        return 0.
    f1 = 2. * (precision * recall) / (precision + recall)
    return f1



