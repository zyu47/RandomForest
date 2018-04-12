try:
    from .node import Node
except ImportError:
    from node import Node

import random


class Tree:
    def __init__(self):
        self.head = None

    def read_tree(self):
        pass

    def build_tree(self, samples):
        self.head = Node()
        inds = random.sample(range(samples.shape[0]), samples.shape[0])  # shuffled index
        for i in inds:
            self.head.add(samples[i])

    def add_bulk(self, samples):
        inds = random.sample(range(samples.shape[0]), samples.shape[0])  # shuffled index
        for i in inds:
            self.head.add(samples[i])

    def find_nn_label(self, sample):
        return self.head.find_nearest_neighbor(sample)

    def print_tree(self):
        '''
        For testing purpose only
        '''
        self.head.print_node()

    def trace(self, sample):
        self.head.trace(sample)
