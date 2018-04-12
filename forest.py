try:
    from .tree import Tree
except ImportError:
    from tree import Tree

import random


class Forest:
    def __init__(self):
        self.trees = []  # trees in forest
        # print('Test')

    def read_forest(self):
        pass

    def build_forest(self, samples, n_trees=50):
        for i in range(n_trees):
            # print('Building Tree ' + str(i))
            t = Tree()
            t.build_tree(samples)
            self.trees.append(t)

    def add_bulk(self, samples):
        for t in self.trees:
            t.add_bulk(samples)

    def find_nn(self, sample, method=0):
        best_label = None
        res = []
        if method == 0:
            # majority voting
            count = {}
            # best_label = None
            max_cnt = 0
            # res = []
            for t in self.trees:
                label, dist = t.find_nn_label(sample)
                if label in count:
                    count[label] += 1
                else:
                    count[label] = 1
                if count[label] > max_cnt:
                    best_label = label
                    max_cnt = count[label]
                res.append((label, dist))
            # print('\t', count)
        elif method == 1:
            # closest label
            closest_dist = 2
            for t in self.trees:
                label, dist = t.find_nn_label(sample)
                if dist < closest_dist:
                    best_label = label
                    closest_dist = dist
                res.append((label, dist))

        return best_label, res

    def trace(self, sample, tree_no=0):
    ## testing purpose only
        self.trees[tree_no].trace(sample)

    def print_forest(self, tree_no=0):
        '''
        For testing purpose only
        :param tree_no: Choose which tree to print; if -1, print all trees
        '''
        if tree_no == -1:
            for t in self.trees:
                t.print_tree()
        else:
            self.trees[tree_no].print_tree()
