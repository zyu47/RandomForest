import os

import numpy as np

import solver
import preprocess

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
        # get training samples
        training_samples = None
        training_labels = None
        for k in self.all_labels_by_subject_id:
            if k == leave_subject:
                continue
            if training_samples is None:
                training_samples = self.all_samples_by_subject_id[k]
            else:
                training_samples = np.concatenate((training_samples, self.all_samples_by_subject_id[k]), axis=0)
            if training_labels is None:
                training_labels = self.all_labels_by_subject_id[k]
            else:
                training_labels = np.concatenate((training_labels, self.all_labels_by_subject_id[k]), axis=0)

        for epoch in range(max_epoch):
            for i in range(5):
                print('*'*30)
            print('EPOCH %d' % epoch)
            for i in range(5):
                print('*'*30)
            self.augmentation = self._get_new_augmentation()
            self.batch_loss = []
            for xbatch, ybatch in self._create_batch(training_samples, training_labels):
                loss = self.solver_instance.train_step(xbatch, ybatch, self.learning_rate)
                self.batch_loss.append(loss)
            self.epoch_losses.append(np.mean(self.batch_loss))
            self._decay_lr()

    def _create_batch(self, training_samples, training_labels):
        num_samples = len(training_samples)
        training_indices = list(range(num_samples))
        np.random.shuffle(training_indices)
        for i in range((num_samples - 1) // self.solver_hps.batch_size + 1):
            start = i*self.solver_hps.batch_size
            end = min((i+1)*self.solver_hps.batch_size, num_samples)
            training_ybatch = training_labels[training_indices[start:end]]
            training_xbatch = []
            for j in range(start, end):
                file_name, offline_aug = training_samples[training_indices[j]]
                # print(offline_aug)
                video = self.primary_process.read_both_channels(file_name)
                video = self.augmentation.offline_aug(video, int(offline_aug))
                if np.random.choice([0, 1]) == 1:
                    # augmentation
                    video = self.augmentation.online_aug(video)
                training_xbatch.append(video)

            yield training_xbatch, training_ybatch

    def _decay_lr(self):
        if len(self.epoch_losses) < 40:
            return
        if self.epoch_losses[-1] / self.epoch_losses[-40] > 0.9:
            self.learning_rate /= 2
            self.num_lr_decay += 1

    def _get_new_augmentation(self):
        return preprocess.Augmentation(self.preprocess_hps)

if __name__ == '__main__':
    train = Training()
    train.run(5)
