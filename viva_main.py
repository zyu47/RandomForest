import os
import numpy as np

import random_forest_viva.preprocess as preprocess
import random_forest_viva.solver as solver
from forest import Forest

os.environ["CUDA_VISIBLE_DEVICES"]="0"


class RandomForestViva:
    def __init__(self, leave_subject_ids):
        self.leave_subject_ids = leave_subject_ids

        # get sample names and labels
        self.pl = preprocess.ProcessLabel(depth_path='./random_forest_viva/data/videos/depth')
        #self.pl.all_samples_by_subject_id
        self.all_labels_by_subject_id = {}
        self.all_videos_by_subject_id = {}
        self.all_features_by_subject_id = {}

        self.primary_process = preprocess.PrimaryProcess(final_depth=32, final_img_shape=(57, 125),
                                                         intensity_vid_root='./random_forest_viva/data/videos',
                                                         depth_vid_root='./random_forest_viva/data/videos/depth')
        # first read all videos into memory
        self._read_all_videos()

        # model-based attributes
        self.solver_hps = solver.HParams(model_type='lrn',
                                         log_path='random_forest_viva/result/test_on_8',
                                         batch_size=40,
                                         weight_decay=0.005,
                                         lr_decay=2,
                                         momentum=0.9,
                                         keep_prob=0.5,
                                         is_loading_model=True)
        self.solver = solver.Solver(self.solver_hps)

    def run(self):
        for leave_subject in self.leave_subject_ids:
            self.solver.load_model('random_forest_viva/result/test_on_' + str(leave_subject))
            acc = self._get_features()
            self._write_result('Neural net accuracy tested on subject %d: %.2f%%' % (leave_subject, acc * 100))
            # split to train/test
            trainng_features = []
            training_labels = []
            for k in self.all_features_by_subject_id:
                if k == leave_subject:
                    continue
                trainng_features += self.all_features_by_subject_id[k].tolist()
                training_labels += np.argmax(self.all_labels_by_subject_id[k], axis=1).tolist()
            # build forest
            forest = Forest()
            forest.build_forest(trainng_features, training_labels, n_trees=25)
            predicted_labels, _ = forest.find_nn(self.all_features_by_subject_id[leave_subject])
            # print(predicted_labels, np.argmax(self.all_labels_by_subject_id[leave_subject], axis=1))
            forest_pred_acc = self._get_forest_acc(predicted_labels,
                                                   np.argmax(self.all_labels_by_subject_id[leave_subject], axis=1))
            self._write_result('Forest accuracy tested on subject %d: %.2f%%' % (leave_subject, forest_pred_acc*100))

    def _get_features(self):
        """
        Retrieve all features for a particular model
        :return:
        """
        nn_acc_all = []
        for k in self.all_videos_by_subject_id:
            acc, self.all_features_by_subject_id[k] =\
                self.solver.sess.run([self.solver.model.accuracy, self.solver.model.features],
                                     {self.solver.model.input: self.all_videos_by_subject_id[k],
                                      self.solver.model.labels: self.all_labels_by_subject_id[k],
                                      self.solver.model.keep_prob: 1.0})
            nn_acc_all.append(acc)
        acc = np.sum(np.array(nn_acc_all) *\
              np.array([len(self.all_labels_by_subject_id[k]) for k in self.all_labels_by_subject_id]))
        acc /= np.sum([len(self.all_labels_by_subject_id[k]) for k in self.all_labels_by_subject_id])

        print('Retrieved all features')
        return acc

    def _get_forest_acc(self, pre_labels, true_labels):
        return np.mean(np.equal(pre_labels, true_labels).astype(np.float32))

    def _write_result(self, content):
        f = open('./log.txt', 'a+')
        f.write(content)
        f.close()

    def _read_all_videos(self):
        # test
        # training_sample_names = training_sample_names[:50]
        # testing_sample_names = testing_sample_names[:40]
        # testing_labels = testing_labels[:40]
        # read all training videos
        for k in self.pl.all_samples_by_subject_id:
            print('Reading video from subject id %d' %k)
            self.all_videos_by_subject_id[k] = []
            self.all_labels_by_subject_id[k] = \
                self.pl.all_labels_by_subject_id[k][0:len(self.pl.all_labels_by_subject_id[k]):preprocess.off_line_aug_num]
            # print(self.all_labels_by_subject_id[k])
            for file_name_ind in range(0, len(self.pl.all_samples_by_subject_id[k]), preprocess.off_line_aug_num):
                # print(self.pl.all_samples_by_subject_id[k][file_name_ind])
                self.all_videos_by_subject_id[k].append(
                    self.primary_process.read_both_channels(self.pl.all_samples_by_subject_id[k][file_name_ind]))
            # print(np.argmax(self.all_labels_by_subject_id[k], axis=1))
            # print(np.array(self.all_videos_by_subject_id[k]).shape)

if __name__ == '__main__':
    t = RandomForestViva([1,2,3,4,5,6,7,8])
    t.run()