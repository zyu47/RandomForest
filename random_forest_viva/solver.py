import os
from collections import namedtuple

import tensorflow as tf

try:
    from model import Model
except:
    from .model import Model


HParams = namedtuple("HParams",
                     ["model_type", "log_path",
                      "batch_size", "weight_decay",
                      "lr_decay", "momentum", "keep_prob",
                      "is_loading_model"])


class Solver:
    def __init__(self, hps):
        self.hps = hps

        # summary and model path
        self.train_log_path = os.path.join(self.hps.log_path, 'train', self.hps.model_type)
        self.val_log_path = os.path.join(self.hps.log_path, 'val', self.hps.model_type)
        self.model_path = os.path.join(self.hps.log_path, 'model', self.hps.model_type, 'model.ckpt')
        for p in [self.train_log_path, self.val_log_path, self.model_path]:
            if not os.path.exists(os.path.dirname(p)):
                os.makedirs(os.path.dirname(p))

        # create model and session
        self.model = Model(self.hps.batch_size, self.hps.weight_decay, self.hps.model_type)
        self.sess = tf.Session()

        # training parameters
        self.lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.momentum = tf.placeholder(tf.float32, shape=[], name="momentum")
        self.global_step = tf.Variable(0, name="global_step")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        # summaries
        tf.summary.scalar("learning rate", self.lr)
        tf.summary.scalar("loss", self.model.loss)
        tf.summary.scalar("accuracy", self.model.accuracy)
        self.summaries = tf.summary.merge_all()

        # model saver
        self.saver = tf.train.Saver()

        # training steps
        self.train_op = self.optimizer.minimize(self.model.loss, global_step=self.global_step)
        self.train_writer = tf.summary.FileWriter(self.train_log_path, self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.val_log_path, self.sess.graph)

        if not self.hps.is_loading_model:
            self.sess.run(tf.global_variables_initializer())

    def train_step(self, xbatch, ybatch, learning_rate):
        _, summary, loss, acc, step = self.sess.run(
            fetches=[self.train_op, self.summaries, self.model.loss,
                     self.model.accuracy, self.global_step],
            feed_dict={self.model.input: xbatch,
                       self.model.labels: ybatch,
                       self.model.keep_prob: self.hps.keep_prob,
                       self.lr: learning_rate,
                       self.momentum: self.hps.momentum})
        print("Training step %d: loss - %.2f; acc - %.2f%%" % (step, loss, acc*100))
        self.train_writer.add_summary(summary, step)

        return loss

    def val_step(self, xbatch, ybatch, learning_rate):
        summary, loss, acc, step = self.sess.run(
            fetches=[self.summaries, self.model.loss,
                     self.model.accuracy, self.global_step],
            feed_dict={self.model.input: xbatch,
                       self.model.labels: ybatch,
                       self.model.keep_prob: 1.0,
                       self.lr: learning_rate})
        print("Validation step %d: loss - %.2f; acc - %.2f%%" % (step, loss, acc * 100))
        self.val_writer.add_summary(summary, step)

        return acc, loss

    def predict(self, xbatch):
        return self.sess.run(
            fetches=[self.model.predicted_labels],
            feed_dict={self.model.input: xbatch,
                       self.model.keep_prob: 1.0})

    def save_model(self):
        self.saver.save(self.sess, self.model_path, self.global_step)

    def load_model(self, log_path):
        self.train_log_path = os.path.join(log_path, 'train', self.hps.model_type)
        self.val_log_path = os.path.join(log_path, 'val', self.hps.model_type)
        self.model_path = tf.train.latest_checkpoint(os.path.join(log_path, 'model', self.hps.model_type))
        print(self.model_path)
        self.saver.restore(self.sess, self.model_path)

    # def get_features(self, input):
    #     return self.sess.run(self.model.features, {self.model.input: input,
    #                                                self.model.keep_prob: 1.0})

if __name__ == '__main__':
    hps = HParams(model_type='lrn',
                  log_path='result',
                  batch_size=40,
                  weight_decay=0.005,
                  lr_decay=2,
                  momentum=0.9,
                  is_loading_model=False,
                  keep_prob=0.5)
    s = Solver(hps)
