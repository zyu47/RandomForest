from tree import Tree


class Forest:
    def __init__(self):
        self.trees = []  # trees in forest

    def read_forest(self):
        pass

    def build_forest(self, samples, n_trees=50):
        for i in range(n_trees):
            t = Tree()
            t.build_tree(samples)
            self.trees.append(t)

    def find_nn(self):
        pass
