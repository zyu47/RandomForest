from node import Node


class Tree:
    def __init__(self):
        self.head = None

    def read_tree(self):
        pass

    def build_tree(self, samples):
        self.head = Node()
        for s in samples:
            self.head.add(s)

    def find_nn_label(self, sample):
        return self.head.find_nearest_neighbor()
