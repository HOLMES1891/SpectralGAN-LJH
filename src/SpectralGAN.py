import os
import numpy as np
import tensorflow as tf
import config
import generator
import discriminator
import random
import load_data
import utils
import test
from scipy.sparse import linalg
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"]= '0'
data = load_data.Data(train_file=config.train_filename, test_file=config.test_filename)


class SpectralGAN(object):
    def __init__(self, n_users, n_items, R):
        print("reading graphs...")
        self.n_users, self.n_items = n_users, n_items
        self.n_node = n_users + n_items
        # construct graph
        self.R = R
        self.adj = self.adj_mat(R=R)
        self.eigenvalues, self.eigenvectors = linalg.eigs(self.adj, k=config.n_eigs)

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

    def build_generator(self):
        """initializing the generator"""

        with tf.variable_scope("generator"):
            self.generator = generator.Generator(n_node=self.n_node, n_layer=config.n_layers)

    def build_discriminator(self):
        """initializing the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(data=data, n_node=self.n_node, n_layer=config.n_layers)

    def train(self):

        print("start training...")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)
            # D-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            labels = []
            losess = []

            all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.eigen_vectors: self.eigenvectors,
                                                                           self.generator.eigen_values: self.eigenvalues})
            for d_epoch in range(config.n_epochs_dis):
                d_s = time()
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % config.dis_interval == 0:
                    node_1, node_2, labels = self.prepare_data_for_d(all_score)
                p_e = time()

                _, loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss],
                              feed_dict={self.discriminator.eigen_vectors: self.eigenvectors,
                                         self.discriminator.eigen_values: self.eigenvalues,
                                         self.discriminator.node_id: np.array(node_1),
                                         self.discriminator.node_neighbor_id: np.array(node_2),
                                         self.discriminator.label: np.array(labels)})
                losess.append(loss)
                d_e = time()
                # print("depoch {} prepare for d {}, update {}".format(d_epoch, p_e-d_s, d_e-p_e))
            print("d_loss %f" % np.mean(np.asarray(losess)))

            # G-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            reward = []
            losess = []
            for g_epoch in range(config.n_epochs_gen):
                g_s = time()
                if g_epoch % config.gen_interval == 0:
                    node_1, node_2, reward = self.prepare_data_for_g()
                p_e = time()
                _, loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                feed_dict={self.generator.eigen_vectors: self.eigenvectors,
                                           self.generator.eigen_values: self.eigenvalues,
                                           self.generator.node_id: np.array(node_1),
                                           self.generator.node_neighbor_id: np.array(node_2),
                                           self.generator.reward: np.array(reward)})
                g_e = time()
                losess.append(loss)
                # print("gepoch {} prepare for d {}, update {}".format(g_epoch, p_e-g_s, g_e - p_e))

            print("g_loss %f" % np.mean(np.asarray(losess)))
            ret = test.test(sess=self.sess, model=self.discriminator, users_to_test=data.test_set.keys())
            print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
                  % (ret[0], ret[1], ret[2], ret[3], ret[4]))
            print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
                  % (ret[5], ret[6], ret[7], ret[8], ret[9]))
        print("training completes")

    def prepare_data_for_d(self, all_score):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        users = random.sample(range(self.n_users), config.missing_edge)

        pos_items = []
        negative_items = []

        for u in users:
            p_items = set(np.nonzero(self.R[u, :])[0].tolist())
            p_item = random.sample(p_items, 1)[0]
            pos_items.append(p_item)
            neg_items = list(set(range(self.n_items)) - set(np.nonzero(self.R[u, :])[0].tolist()))
            relevance_probability = all_score[u, neg_items]
            relevance_probability = utils.softmax(relevance_probability)
            neg_item = np.random.choice(neg_items, size=1, p=relevance_probability)[0]  # select next node
            negative_items.append(neg_item)

        node_2 = [self.n_users + p for p in pos_items] + [self.n_users + p for p in negative_items]
        node_1 = users*2
        labels = [1.0]*config.missing_edge + [0.0] * config.missing_edge

        return node_1, node_2, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        users = random.sample(range(self.n_users), config.missing_edge)

        all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.eigen_vectors: self.eigenvectors,
                                                                       self.generator.eigen_values: self.eigenvalues})

        negative_items = []
        for u in users:
            pos_items = set(np.nonzero(self.R[u, :])[0].tolist())
            neg_items = list(set(range(self.n_items)) - pos_items)
            relevance_probability = all_score[u, neg_items]
            relevance_probability = utils.softmax(relevance_probability)
            neg_item = np.random.choice(neg_items, size=1, p=relevance_probability)[0]  # select next node
            negative_items.append(data.n_users + neg_item)

        node_1 = users*2
        node_2 = negative_items*2
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.eigen_vectors: self.eigenvectors,
                                          self.discriminator.eigen_values: self.eigenvalues,
                                          self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1[:config.missing_edge], node_2[:config.missing_edge], reward[:config.missing_edge]

    def adj_mat(self, R, self_connection=True):
        A = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items], dtype=np.float32)
        A[:self.n_users, self.n_users:] = R
        A[self.n_users:, :self.n_users] = R.T
        if self_connection:
            return np.identity(self.n_users + self.n_items, dtype=np.float32) + A
        return A


if __name__ == "__main__":
    spectral_gan = SpectralGAN(n_users=data.n_users, n_items=data.n_items, R=data.R)
    spectral_gan.train()
