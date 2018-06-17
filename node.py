try:
    from . import parameters
except ImportError:
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

    def add(self, sample, label):
        if self._internal_node:  # if this node is already split
            if self._distance_metric(sample, self._items[0][0]) <= self._dist_threshold:
                self._left_child.add(sample, label)
            else:
                self._right_child.add(sample, label)
        else:
            # first check if sample already exists; if so, do not add sample into tree again
            for i in self._items:
                if np.array_equal(i[0], sample):
                    return
            self._items.append((sample, label))
            if len(self._items) > self._item_cnt_cap and self._split_condition_met():
                self._split_node()

    def find_nearest_neighbor(self, sample):
        if self._internal_node:
            if self._distance_metric(sample, self._items[0][0]) <= self._dist_threshold:
                return self._left_child.find_nearest_neighbor(sample)
            else:
                return self._right_child.find_nearest_neighbor(sample)
        else:
            if len(self._items) == 0:
                raise RuntimeError('No matching label in database!')
            distances = [self._distance_metric(sample, i[0]) for i in self._items]
            return self._items[np.argmin(distances)][1], np.min(distances)

    def _split_node(self):
        self._left_child = Node()
        self._right_child = Node()

        # pick a random pivot
        pivot = self._items[random.randint(0, len(self._items) - 1)]

        # calculate all the distances and choose median distance as splitting threshold
        distances = [self._distance_metric(i[0], pivot[0]) for i in self._items]
        self._dist_threshold = np.median(distances)

        for i in self._items:
            if self._distance_metric(i[0], pivot[0]) <= self._dist_threshold:
                self._left_child.add(i[0], i[1])
            else:
                self._right_child.add(i[0], i[1])

        # save pivot information
        self._items = [pivot]

        # change node status
        self._internal_node = True

    def _split_condition_met(self):
        if self._split_method == 0:
            return True

    def _distance_metric(self, a, b):
        """
        :param a: One item
        :param b: Another item
        :return: Distance between items a and b
        """
        if self._distance_metric_type == 0:
            return -pearsonr(a.flatten(), b.flatten())[0]  # to match with euclidean distance measure, negate
        elif self._distance_metric_type == 1:
            return np.linalg.norm(a.flatten()-b.flatten())
        elif self._distance_metric_type == 2:
            return -np.sum(a.flatten() * b.flatten())
        else:
            raise ValueError('Unrecognized distance metric type')

    def print_node(self, level=0):
        """
        For testing purpose only
        """
        if self._internal_node:
            print('-'*level, self._dist_threshold)
            self._left_child.print_node(level+1)
            self._right_child.print_node(level+1)
        else:
            print('-'*level, len(self._items), '*')

    def trace(self, sample, level=0):
        """
        For testing purpose only
        """
        if self._internal_node:
            if self._distance_metric(sample, self._items[0][0]) <= self._dist_threshold:
                print('-'*level,
                      'GoTo left',
                      'Threshold: ' + str(self._dist_threshold),
                      'Dist: ' + str(self._distance_metric(sample, self._items[0][0])))
                return self._left_child.trace(sample)
            else:
                print('-'*level,
                      'GoTo right',
                      'Threshold: ' + str(self._dist_threshold),
                      'Dist: ' + str(self._distance_metric(sample, self._items[0][0])))
                return self._right_child.trace(sample)
        else:
            # distances = [self._distance_metric(sample, i) for i in self._items]
            # return self._items[np.argmin(distances)][1]
            return self._items

