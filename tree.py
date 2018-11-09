try:
    from node import Node
except:
    from .node import Node

import random


class Tree:
    def __init__(self, item_per_node=None):
        self.head = None
        self.item_per_node = item_per_node

    def read_tree(self):
        pass

    def build_tree(self, samples, labels):
        self.head = Node(self.item_per_node)
        inds = random.sample(range(samples.shape[0]), samples.shape[0])  # shuffled index
        for i in inds:
            self.head.add(samples[i], labels[i])

    def add_new(self, samples, labels):
        inds = random.sample(range(samples.shape[0]), samples.shape[0])  # shuffled index
        for i in inds:
            self.head.add(samples[i], labels[i])

    def find_nn_label(self, sample):
        return self.head.find_nearest_neighbor(sample)

    def print_tree(self):
        '''
        For testing purpose only
        '''
        self.head.print_node()

    def trace(self, sample):
        '''
        For testing purpose only
        '''
        self.head.trace(sample)
