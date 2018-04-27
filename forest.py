try:
    from .tree import Tree
except ImportError:
    from tree import Tree
import numpy as np

frame_per_clip = 15
feature_dim = 1024


class Forest:
    def __init__(self):
        self.trees = []  # trees in forest
        # print('Test')

    def read_forest(self):
        pass

    def build_forest(self, samples, labels, n_trees=50):
        '''
        :param samples: Training samples to build trees, with dimension: nSamples*15*1024
        :param labels:  Labels of training samples, with dimension: nSamples
        :param n_trees: Number of trees to be built, default to 50 trees
        :return:        None
        '''

        # first normalize the samples
        samples_norm = self._normalize_sample(samples)

        for i in range(n_trees):
            print('Building Tree #' + str(i))
            t = Tree()
            t.build_tree(samples_norm, labels)
            self.trees.append(t)

    def add_new(self, samples, labels):
        '''
        This function is used to add a GROUP of or a SINGLE new samples into forest
        :param samples: New training samples to build trees, with dimension: nSamples*15*1024 or 15*1024
        :param labels:  Labels of new samples, with dimension: nSamples
        :return:        None
        '''

        # first normalize the samples
        samples_norm = self._normalize_sample(samples)

        for t in self.trees:
            t.add_new(samples_norm, labels)

    def find_nn(self, samples, method=1):
        '''
        :param samples: A set of samples that need to find the labels
        :param method:  Either use majority voting (0) or return the closest sample label
        :return:        The labels found and the result from each tree
        '''
        samples_norm = self._normalize_sample(samples)
        best_labels = [-2 for i in range(samples_norm.shape[0])]
        details = []
        for i, s in enumerate(samples_norm):
            details.append([])
            if method == 0:  # majority voting
                count = {}
                max_cnt = 0
                for t in self.trees:
                    label, dist = t.find_nn_label(s)
                    if label in count:
                        count[label] += 1
                    else:
                        count[label] = 1
                    if count[label] > max_cnt:
                        best_labels[i] = label
                        max_cnt = count[label]
                    details[-1].append((label, dist))
                # print('\t', count)
            elif method == 1:  # closest sample label
                closest_dist = feature_dim*frame_per_clip + 1  # theoretical upper bound of distance
                for t in self.trees:
                    label, dist = t.find_nn_label(s)
                    if dist < closest_dist:
                        best_labels[i] = label
                        closest_dist = dist
                    details[-1].append((label, dist))

        return best_labels, details

    def _normalize_sample(self, samples):
        samples_norm = np.copy(samples)
        if len(samples_norm.shape) == 2:  # only one sample, add one extra dimension to dim 0
            samples_norm = samples_norm[np.newaxis, :, :]
        for i in range(samples_norm.shape[0]):
            samples_norm[i] -= np.mean(samples_norm[i])
            samples_norm[i] /= np.std(samples_norm[i])

        return samples_norm

    def trace(self, sample, tree_no=0):
        ## testing purpose only
        s = self._normalize_sample(sample)
        self.trees[tree_no].trace(sample[0])

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
