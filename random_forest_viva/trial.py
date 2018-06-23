# import tensorflow as tf
#
# x = tf.Variable([[1],[2],[3],[4],[5]])
# y = tf.ones(tf.shape(x)[-1])
#
# sess = tf.Session()
# with sess.as_default():
#     sess.run(tf.global_variables_initializer())
#     print(y.eval())
# import cv2
# print(cv2.CV_32F)
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import shift

# def dropout(v_in):
#     y = np.random.choice([True, False], (3,2,3))
#     v_in[y] = 0
#
# x = np.ones((3,2, 3))
#
# print(len([1,2,3,4,6,7,8,13,14,15,16,21,23,27,28,29,30,31,32]))
print(len(list(range(0, 3088, 4))))