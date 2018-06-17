import tensorflow as tf


class Model:
    """
    Implements the 3D convolution model for VIVA challenge (77.5% accuracy from NVIDIA)
    """
    def __init__(self, batch_size, model_type='lrn'):
        self.batch_size = batch_size
        self.input = tf.placeholder(tf.float32,
                                    shape=[self.batch_size, 57, 125, 64],
                                    name="input")
        self.labels = tf.placeholder(tf.int8,
                                     shape=[self.batch_size, 19],
                                     name="output")
        if model_type == 'lrn':
            self.x = tf.image.resize_nearest_neighbor(self.input, [28, 62])


    def _build_model(self):

    def __conv_pool_layer(self, kernel, pool, ind, input_matrix):
        in_shape = tf.shape(input_matrix)
        n_in = in_shape[1] * in_shape[2] * in_shape[3]  # number of input neurons
        n_out = in_shape[1] * in_shape[2] * in_shape[3]  # number of output neurons
        w = tf.Variable(tf.random_)

    def __fc_layer(self, out_dim, ind, input_matrix):

