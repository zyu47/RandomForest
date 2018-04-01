import parameters
import random
import numpy as np
from scipy.stats import pearsonr


class Node:
    def __init__(self):
        self._left_child = None
        self._right_child = None
        self._internal_node = False  # Indicate this node is a splitting node or collecting node
        self._item_cnt_cap = parameters.item_cnt_cap  # The maximum number of items in a node before splitting
        self._split_method = parameters.split_mehod  # The choice of splitting criteria
        self._distance_metric_type = parameters.distance_metric_type  # The type of distance metric.

        self._items = []  # Note that each item has first element as vector and second as label
        self._dist_threshold = 0  # Any node with smaller distance to pivot than dist_threshold goes to left child

    def add(self, sample):
        if self._internal_node:  # if this node is already split
            if self._distance_metric(sample, self._items[0]) <= self._dist_threshold:
                self._left_child.add(sample)
            else:
                self._right_child.add(sample)
        else:
            self._items.append(sample)
            if len(self._items) > self._item_cnt_cap and self._split_condition_met():
                self._split_node()

    def find_nearest_neighbor(self, sample):
        if self._internal_node:
            if self._distance_metric(sample, self._items[0]) <= self._dist_threshold:
                return self._left_child.find_nearest_neighbor(sample)
            else:
                return self._right_child.find_nearest_neighbor(sample)
        else:
            distances = [self._distance_metric(sample, self._items[0]) for i in self._items]
            return self._items[np.argmin(distances)][1]

    def _split_node(self):
        self._left_child = Node()
        self._right_child = Node()

        # pick a random pivot
        pivot = self._items[random.randint(0, len(self._items) - 1)]

        # calculate all the distances and choose median distance as splitting threshold
        distances = [self._distance_metric(i, pivot) for i in self._items]
        self._dist_threshold = np.median(distances)

        for i in self._items:
            if self._distance_metric(i, pivot) <= self._dist_threshold:
                self._left_child.add(i)
            else:
                self._right_child.add(i)

        # save pivot information
        self._items = [pivot]

    def _split_condition_met(self):
        if self._split_method == 0:
            return True

    def _distance_metric(self, a, b):
        '''
        :param a: One item
        :param b: Another item
        :return: Distance between items a and b
        '''
        if self._distance_metric_type == 0:
            return pearsonr(a[0].flatten(), b[0].flatten())
        elif self._distance_metric_type == 1:
            return np.linalg.norm(a[0].flatten(), b[0].flatten())
        else:
            raise ValueError('Unrecognized distance metric type')
