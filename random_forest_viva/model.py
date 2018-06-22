import tensorflow as tf


class Model:
    """
    Implements the 3D convolution model for VIVA challenge (77.5% accuracy from NVIDIA)
    Input shape: [batch_size = 20/40, depth = 32, width = 57, height = 125, channel = 2]
    """
    def __init__(self, batch_size, l2_reg, model_type='lrn'):
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.input = tf.placeholder(tf.float32,
                                    shape=[None, 32, 57, 125, 2],
                                    name="input")
        self.labels = tf.placeholder(tf.int8,
                                     shape=[None, 19],
                                     name="output")
        self.keep_prob = tf.placeholder(tf.float32,
                                       shape=[],
                                       name="keep_probability")
        self.parameters = {'conv': None, 'fc': [512, 256, 19]}
        self.features = None
        self.loss_raw = None
        self.regularizers = 0
        self.loss = None  # loss with regularization
        
        self.model_type = model_type
        if model_type == 'lrn':
            self.x = tf.reshape(self.input, [-1, 57, 125, 2])
            self.x = tf.image.resize_nearest_neighbor(self.x, [28, 62])
            self.x = tf.reshape(self.x, [-1, 32, 28, 62, 2])
            # convolution filters and pooling filters shape
            self.parameters['conv'] = [[(5, 5, 5, 2, 8), (1, 2, 2, 2, 1)],
                                       [(3, 5, 5, 8, 32), (1, 2, 2, 2, 1)],
                                       [(3, 3, 5, 32, 64), (1, 1, 1, 4, 1)]]
        else:
            raise ValueError("HRN not implemented yet")

        self.__build_conv_model(self.parameters, self.x)

    def __build_conv_model(self, parameters, input_matrix):
        result = input_matrix

        # building convolutional part
        for ind, shapes in enumerate(parameters['conv']):
            result = self.__conv_pool_layer(shapes[0], shapes[1], ind, result)

        # flatten convolutional result
        if self.model_type == 'lrn':
            result = tf.reshape(result, [-1, 64*2*2*4])
        else:
            raise ValueError('HRN not implemented yet')

        # fully-connected layer
        for ind, shapes in enumerate(parameters['fc'][:-1]):
            result = self.__fc_layer(shapes, ind, False, result)

        # softmax layer and loss
        self.features = result  # for extracting features
        logits = self.__fc_layer(parameters['fc'][-1], -1, True, result)
        self.loss_raw = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                logits=logits))
        self.loss = self.loss_raw + self.l2_reg * self.regularizers / 2

        # predicted labels
        self.predicted_labels = tf.argmax(logits, 1)
        correct_prediction = tf.cast(tf.equal(self.predicted_labels, tf.argmax(self.labels, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

    def __conv_pool_layer(self, kernel_shape, pool_shape, ind, input_matrix):
        # 3D convolution
        w = tf.get_variable(name="conv_w_%d" %ind, shape=kernel_shape,
                            initializer=tf.contrib.layers.xavier_initializer())
        self.regularizers += tf.nn.l2_loss(w)
        b = tf.Variable(tf.ones(kernel_shape[-1]), name="conv_b_%d" %ind)
        conv = tf.nn.conv3d(input=input_matrix,
                            filter=w,
                            strides=[1, 1, 1, 1, 1],
                            padding='VALID')
        conv_relu = tf.nn.relu(conv + b)

        # max pooling
        return tf.nn.max_pool3d(input=conv_relu,
                                ksize=pool_shape,
                                strides=pool_shape,
                                padding='SAME')

    def __fc_layer(self, out_dim, ind, pre_softmax, input_matrix):
        w = tf.get_variable(name="fc_w_%d" %ind, shape=[input_matrix.get_shape()[1], out_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        self.regularizers += tf.nn.l2_loss(w)
        if pre_softmax:
            b = tf.Variable(tf.zeros(out_dim), name="fc_b_%d" %ind)
        else:
            b = tf.Variable(tf.ones(out_dim), name="fc_b_%d" % ind)
        fc = tf.nn.xw_plus_b(input_matrix, w, b)

        if pre_softmax:
            return tf.nn.relu(fc)
        else:
            return tf.nn.dropout(tf.nn.relu(fc), self.keep_prob)

if __name__ == '__main__':
    m = Model(batch_size=20, l2_reg=0.005)

