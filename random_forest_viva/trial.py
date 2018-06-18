# import tensorflow as tf
#
# x = tf.Variable([[1],[2],[3],[4],[5]])
# y = tf.ones(tf.shape(x)[-1])
#
# sess = tf.Session()
# with sess.as_default():
#     sess.run(tf.global_variables_initializer())
#     print(y.eval())

x = list(range(40))
print(x[-39])