import tensorflow as tf
import numpy as np


class gen(object):
    def __init__(self):

        self.a = tf.placeholder(tf.int32)
        self.r = 2 * self.a


class test(object):
    def __init__(self):

        self.build_test()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

    def build_test(self):
        self.g = gen()

    def train(self):
        r1 = self.sess.run(self.g.r, feed_dict={self.g.a: np.array(3)})
        print(r1)
        r2 = self.sess.run(self.g.r, feed_dict={self.g.a: np.array([2, 3])})
        print(r2)


if __name__ == "__main__":
    t = test()
    t.train()
