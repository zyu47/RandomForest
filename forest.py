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

    def find_nn(self, sample):
        # majority voting
        count = {}
        best_label = None
        max_cnt = 0
        for t in self.trees:
            label = t.find_nn_label(sample)
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
            if count[label] > max_cnt:
                best_label = label
                max_cnt = count[label]

        return best_label
