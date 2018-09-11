import os
import sys
from threading import Thread

import numpy as np
import queue

import solver
import preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class TFWorker(Thread):
    def __init__(self, queue, solver_hps, leave_subject, testing_videos, testing_labels):
        Thread.__init__(self)
        self.leave_subject = leave_subject
        self.queue = queue
        self.testing_videos = testing_videos
        self.testing_labels = testing_labels
        self.solver_hps = solver_hps
        self.solver_instance = solver.Solver(self.solver_hps)
        self.solver_instance.load_model('result/test_on_'+str(self.leave_subject))

        self.epoch_losses = []
        self.batch_loss = []
        self.learning_rate = 0.005
        self.num_lr_decay = 0
        self.save_every = 6  # save every 6 * num_queue_putters epochs
        self.test_every = 78  # test on every epoch (78 batches / epoch)

    def run(self):
        global_step = 0
        while True:
            try:
                data_batch = self.queue.get(timeout=30)
            except queue.Empty:
                break

            global_step += 1
            xbatch, ybatch, epoch, worker_ind = data_batch
            print('Epoch: ', epoch, " by worker: ", worker_ind)
            loss = self.solver_instance.train_step(xbatch, ybatch, self.learning_rate)
            self.batch_loss.append(loss)
            # testing
            if global_step % self.test_every == 0:
                final_val_acc, _ = self.solver_instance.val_step(self.testing_videos, self.testing_labels,
                                                                 self.learning_rate)
                self.epoch_losses.append(np.mean(self.batch_loss))
                self.batch_loss = []
            self._decay_lr()
            if epoch == self.save_every:
                self.solver_instance.save_model()
                self.save_every += 6

        self.solver_instance.save_model()
        self._save_to_log(self.solver_instance.val_step(self.testing_videos, self.testing_labels,
                                                        self.learning_rate)[0])

    def _decay_lr(self):
        if len(self.epoch_losses) < 40:
            return
        if self.epoch_losses[-1] / self.epoch_losses[-40] > 0.9:
            self.epoch_losses = []
            self.learning_rate /= 2
            self.num_lr_decay += 1

    def _save_to_log(self, acc):
        f = open('./log.txt', 'a+')
        f.write('Test on subject %d: accuracy - %.2f%%\n' % (self.leave_subject, acc * 100))
        f.close()


class QueuePutterWorker(Thread):
    def __init__(self, queue, max_num_epochs, solver_hps, preprocess_hps, training_names, training_videos,
                 training_labels, worker_ind):
        Thread.__init__(self)
        self.queue = queue
        self.max_num_epochs = max_num_epochs
        self.solver_hps = solver_hps
        self.preprocess_hps = preprocess_hps
        self.augmentation = None

        self.training_names = training_names
        self.training_videos = training_videos
        self.training_labels = training_labels

        self.woker_ind = worker_ind

    def run(self):
        for epoch in range(self.max_num_epochs):
            self.augmentation = self._get_new_augmentation()
            self._create_batch(epoch)

    def _create_batch(self, epoch):
        num_samples = len(self.training_names)
        training_indices = list(range(num_samples))
        np.random.shuffle(training_indices)
        for i in range((num_samples - 1) // self.solver_hps.batch_size + 1):
            start = i * self.solver_hps.batch_size
            end = min((i + 1) * self.solver_hps.batch_size, num_samples)
            training_ybatch = self.training_labels[training_indices[start:end]]
            training_xbatch = []
            for j in range(start, end):
                video_ind = training_indices[j] // preprocess.off_line_aug_num
                offline_aug = training_indices[j] % preprocess.off_line_aug_num
                # print(offline_aug)
                # offline augmentation makes a copy, so no need to explicitly make a copy
                video = self.augmentation.offline_aug(self.training_videos[video_ind],
                                                      int(offline_aug))
                # augmentation
                self.augmentation.online_aug(video)
                training_xbatch.append(video)

            self.queue.put((training_xbatch, training_ybatch, epoch, self.woker_ind))

    def _get_new_augmentation(self):
        return preprocess.Augmentation(self.preprocess_hps)


class Training:
    def __init__(self, leave_subject, queue_capacity, max_epoch, num_queue_putter):
        self.leave_subject = leave_subject
        self.solver_hps = solver.HParams(model_type='lrn',
                                         log_path='result/test_on_' + str(self.leave_subject),
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
        self.primary_process = preprocess.PrimaryProcess(self.preprocess_hps.final_depth,
                                                         self.preprocess_hps.final_img_shape)

        self.batch_q = queue.Queue(maxsize=queue_capacity)
        self.max_epoch = max_epoch
        self.num_queue_putter = num_queue_putter

    def start(self):
        training_names, training_labels, training_raw_videos, testing_videos, testing_labels = self._get_all_samples()

        # start data generation workers
        q_putters = []
        for i in range(self.num_queue_putter):
            q_putters.append(QueuePutterWorker(self.batch_q, self.max_epoch // self.num_queue_putter, self.solver_hps,
                                               self.preprocess_hps, training_names, training_raw_videos,
                                               training_labels, i))
            q_putters[-1].start()

        # start TF thread
        tf_worker = TFWorker(self.batch_q, self.solver_hps, self.leave_subject, testing_videos, testing_labels)
        tf_worker.start()

    def _get_all_samples(self):
        # get training samples
        pl = preprocess.ProcessLabel()
        all_samples_by_subject_id = pl.all_samples_by_subject_id
        all_labels_by_subject_id = pl.all_labels_by_subject_id
        training_sample_names = None
        training_labels = None
        testing_sample_names = None
        testing_labels = None
        for k in all_labels_by_subject_id:
            if k == self.leave_subject:
                testing_sample_names = all_samples_by_subject_id[k]
                testing_labels = all_labels_by_subject_id[k]
                continue
            if training_sample_names is None:
                training_sample_names = all_samples_by_subject_id[k]  # (4*num_samples) of file names
                training_labels = all_labels_by_subject_id[k]  # (4*num_samples) * 19
            else:
                training_sample_names = np.concatenate((training_sample_names,
                                                        all_samples_by_subject_id[k]), axis=0)
                training_labels = np.concatenate((training_labels,
                                                  all_labels_by_subject_id[k]), axis=0)

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

        return training_sample_names, training_labels, training_raw_videos, testing_videos, testing_labels


if __name__ == '__main__':
    train = Training(int(sys.argv[1]), 16, 525, 15)
    train.start()
