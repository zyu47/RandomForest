try:
    from .tree import Tree
except ImportError:
    from tree import Tree
import numpy as np
import pickle
import os

frame_per_clip = 15
feature_dim = 1024


class Forest:
    def __init__(self):
        self.trees = []  # trees in forest
        self.sample_dim = None  # the dimensionality of the samples
        self.next_ind = None  # what will be the next index of novel sample

    @staticmethod
    def load_forest(file_name, load_dir=None):
        """
        A static method to load forest from disk
        :param load_dir:  A different directory to save rather than default:
                          /s/red/a/nobackup/vision/jason/forest
        :param file_name: name of the file to save
        :return:          The loaded instance of Forest
        """
        if load_dir is None:
            load_dir = '/s/red/a/nobackup/vision/jason/forest'

        load_path = os.path.join(load_dir, file_name + '.pickle')
        f = open(load_path, 'rb')
        loaded = pickle.load(f)
        f.close()

        return loaded

    def save_forest(self, file_name, save_dir=None):
        """
        Save forest to disk for loading
        :param save_dir:  A different directory to save rather than default:
                          /s/red/a/nobackup/vision/jason/forest
        :param file_name: name of the file to save
        :return:          None
        """
        if save_dir is None:
            save_dir = '/s/red/a/nobackup/vision/jason/forest'

        save_path = os.path.join(save_dir, file_name + '.pickle')
        f = open(save_path, 'wb')
        pickle.dump(self, f)
        f.close()

    def build_forest(self, samples, labels, n_trees=50, verbose=True):
        """
        :param samples: Training samples to build trees, with dimension: nSamples*feature_dim (every frame),
                        nSamples[*frame_per_clip]*feature_dim (every clip) (nSamples >= 2)
        :param labels:  Labels of training samples, with dimension: nSamples
        :param n_trees: Number of trees to be built, default to 50 trees
        :param verbose: Whether to print progress
        :return:        None
        """

        # first normalize the samples
        self.sample_dim = len(np.array(samples[0]).shape)  # if frame, ==1; if clip, ==2
        samples_norm = self._normalize_sample(samples)

        for i in range(n_trees):
            if verbose:
                print('Building Tree #' + str(i))
            t = Tree()
            t.build_tree(samples_norm, labels)
            self.trees.append(t)

        self.next_ind = np.max(labels) + 10  # set next index to be 10 larger than the maximum index of labels

    def add_new(self, samples, labels=None):
        """
        This function is used to add a GROUP of or a SINGLE new samples into forest
        :param samples: New training samples to build trees, with dimension: nSamples*15*1024 or 15*1024
        :param labels:  Labels of new samples, with dimension: nSamples
        :return:        None
        """

        # first normalize the samples
        samples_norm = self._normalize_sample(samples)

        # if labels are not supplied, use internal next_ind
        if labels is None:
            labels = [self.next_ind] * samples_norm.shape[0]

        for t in self.trees:
            t.add_new(samples_norm, labels)

    def find_nn(self, samples):
        """
        :param samples: A set of samples that need to find their labels
        :return:        The predicted label and distance for each sample
        """
        samples_norm = self._normalize_sample(samples)
        best_labels = [-2] * samples_norm.shape[0]
        # (feature_dim*10)  is used as upper bound of distance, actual distance could be smaller
        closest_dists = [feature_dim*10] * samples_norm.shape[0]
        for i, s in enumerate(samples_norm):
            for t in self.trees:
                label, dist = t.find_nn_label(s)
                if dist < closest_dists[i]:
                    best_labels[i] = label
                    closest_dists[i] = dist
            if best_labels[i] == -2:
                best_labels[i] = 2  # 2 is gesture 'other'

        return best_labels, closest_dists

    def _normalize_sample(self, samples):
        samples_norm = np.copy(samples)
        if len(samples_norm.shape) == self.sample_dim:
            # only one sample, add one extra dimension to dim 0
            samples_norm = np.expand_dims(samples_norm, axis=0)
        for i in range(samples_norm.shape[0]):
            samples_norm[i] -= np.mean(samples_norm[i])
            samples_norm[i] /= np.std(samples_norm[i])

        return samples_norm

    def trace(self, sample, tree_no=0):
        """
        For testing purpose only, trace a particular sample in a tree
        """
        s = self._normalize_sample(sample)
        self.trees[tree_no].trace(s[0])

    def print_forest(self, tree_no=0):
        """
        For testing purpose only
        :param tree_no: Choose which tree to print; if -1, print all trees
        """
        if tree_no == -1:
            for t in self.trees:
                t.print_tree()
        else:
            self.trees[tree_no].print_tree()
