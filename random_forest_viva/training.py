import os

import numpy as np

import solver
import preprocess

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Training:
    def __init__(self, depth_path='./data/videos/depth'):
        self.solver_hps = solver.HParams(model_type='lrn',
                                         log_path='result',
                                         batch_size=40,
                                         weight_decay=0.005,
                                         lr_decay=2,
                                         momentum=0.9,
                                         keep_prob=0.5,
                                         is_loading_model=False)
        self.preprocess_hps = preprocess.HParams(final_depth=32, final_img_shape=(57, 125),
                                                 angle_range=(-10, 10), scaling_range=(-0.3, 0.3),
                                                 x_translate_range=(-8, 8), y_translate_range=(-4, 4),
                                                 sigma=10, alpha=6,
                                                 ted_sigma=4, t_scaling_range=(-0.2, 0.2), t_translate_range=(-4, 4))
        self.solver_instance = solver.Solver(self.solver_hps)
        self.primary_process = preprocess.PrimaryProcess(self.preprocess_hps.final_depth,
                                                         self.preprocess_hps.final_img_shape)
        self.augmentation = None

        self.pl = preprocess.ProcessLabel()
        self.all_samples_by_subject_id = self.pl.all_samples_by_subject_id
        self.all_labels_by_subject_id = self.pl.all_labels_by_subject_id

        self.epoch_losses = []
        self.batch_loss = []
        self.learning_rate = 0.005
        self.num_lr_decay = 0

    def run(self, leave_subject, max_epoch=300):
        # print(self.all_samples_by_subject_id)
        # get training samples
        training_sample_names = None
        training_labels = None
        testing_sample_names = None
        testing_labels = None
        for k in self.all_labels_by_subject_id:
            if k == leave_subject:
                testing_sample_names = self.all_samples_by_subject_id[k]
                testing_labels = self.all_labels_by_subject_id[k]
            if training_sample_names is None:
                training_sample_names = self.all_samples_by_subject_id[k]  # (4*num_samples) of file names
                training_labels = self.all_labels_by_subject_id[k]  # (4*num_samples) * 19
            else:
                training_sample_names = np.concatenate((training_sample_names,
                                                       self.all_samples_by_subject_id[k]), axis=0)
                training_labels = np.concatenate((training_labels,
                                                 self.all_labels_by_subject_id[k]), axis=0)

        # test
        # training_sample_names = training_sample_names[:50]
        # testing_sample_names = testing_sample_names[:40]
        # testing_labels = testing_labels[:40]
        # read all training videos
        training_raw_videos = []
        for file_name_ind in range(0, len(training_sample_names), preprocess.off_line_aug_num):
            # print(file_name_ind)
            training_raw_videos.append(
                self.primary_process.read_both_channels(training_sample_names[file_name_ind]))

        # read all testing videos
        testing_videos = []
        for file_name_ind in range(0, len(testing_sample_names), preprocess.off_line_aug_num):
            testing_videos.append(
                self.primary_process.read_both_channels(testing_sample_names[file_name_ind]))
        print('Reading videos done!')
        testing_labels = testing_labels[0:len(testing_labels):preprocess.off_line_aug_num]

        for epoch in range(max_epoch):
            self.augmentation = self._get_new_augmentation()
            self.batch_loss = []
            for xbatch, ybatch in self._create_batch(training_sample_names,
                                                     training_raw_videos,
                                                     training_labels):
                print('Batch %d' %epoch)
                loss = self.solver_instance.train_step(xbatch, ybatch, self.learning_rate)
                self.batch_loss.append(loss)
            # testing
            self.solver_instance.val_step(testing_videos, testing_labels, self.learning_rate)
            self.epoch_losses.append(np.mean(self.batch_loss))
            self._decay_lr()
            if self.num_lr_decay == 4:
                break

    def _create_batch(self, training_names, training_videos, training_labels):
        num_samples = len(training_names)
        training_indices = list(range(num_samples))
        np.random.shuffle(training_indices)
        for i in range((num_samples - 1) // self.solver_hps.batch_size + 1):
            start = i*self.solver_hps.batch_size
            end = min((i+1)*self.solver_hps.batch_size, num_samples)
            training_ybatch = training_labels[training_indices[start:end]]
            training_xbatch = []
            for j in range(start, end):
                video_ind = training_indices[j] // preprocess.off_line_aug_num
                offline_aug = training_indices[j] % preprocess.off_line_aug_num
                # print(offline_aug)
                # offline augmentation makes a copy, so no need to explicitly make a copy
                video = self.augmentation.offline_aug(training_videos[video_ind],
                                                      int(offline_aug))
                # augmentation
                self.augmentation.online_aug(video)
                training_xbatch.append(video)

            yield training_xbatch, training_ybatch

    def _decay_lr(self):
        if len(self.epoch_losses) < 40:
            return
        if self.epoch_losses[-1] / self.epoch_losses[-40] > 0.9:
            self.epoch_losses = []
            self.learning_rate /= 2
            self.num_lr_decay += 1

    def _get_new_augmentation(self):
        return preprocess.Augmentation(self.preprocess_hps)

if __name__ == '__main__':
    train = Training()
    train.run(5)
