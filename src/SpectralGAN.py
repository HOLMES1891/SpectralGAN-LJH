import os
import numpy as np
import tensorflow as tf
from src import config
from src import generator
from src import discriminator
import random
from src import load_data
from src import utils
from src import test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#   Load data
data = load_data.Data(train_file=config.train_filename, test_file=config.test_filename)


class SpectralGAN(object):
    def __init__(self, n_users, n_items, R):
        print("reading graphs...")
        self.n_users, self.n_items = n_users, n_items
        self.n_node = n_users + n_items
        # construct graph
        self.R = R

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
            self.discriminator = discriminator.Discriminator(n_node=self.n_node, n_layer=config.n_layers)

    def train(self):

        print("start training...")
        for epoch in range(config.n_epochs):
            print("----------- epoch %d -----------" % epoch)
            # D-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            labels = []
            for d_epoch in range(config.n_epochs_dis):
                print("train D epoch {}".format(d_epoch))
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % config.dis_interval == 0:
                    adj_missing, node_1, node_2, labels = self.prepare_data_for_d()

                self.sess.run(self.discriminator.d_updates,
                              feed_dict={self.discriminator.adj_miss: np.array(adj_missing),
                                         self.discriminator.node_id: np.array(node_1),
                                         self.discriminator.node_neighbor_id: np.array(node_2),
                                         self.discriminator.label: np.array(labels)})
            # G-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            reward = []
            for g_epoch in range(config.n_epochs_gen):
                print("train G epoch {} ".format(g_epoch))
                if g_epoch % config.gen_interval == 0:
                    adj_missing, node_1, node_2, reward = self.prepare_data_for_g()

                self.sess.run(self.generator.g_updates,
                                feed_dict={self.generator.adj_miss: np.array(adj_missing),
                                           self.generator.i: np.array(node_1),
                                           self.generator.u: np.array(node_2),
                                           self.generator.reward: np.array(reward)})

            print("begin test...")
            ret = test.test(sess=self.sess, model=self.generator, users_to_test=data.test_set.keys())
            print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
                  % (ret[0], ret[1], ret[2], ret[3], ret[4]))
            print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
                  % (ret[5], ret[6], ret[7], ret[8], ret[9]))
        print("training completes")

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        users = random.sample(range(self.n_users), config.missing_edge)

        R_missing = np.copy(self.R)
        pos_items = []
        node_1 = []

        for u in users:
            """
                这里不删边，以保证每次 GCN 都是在同一张图上进行卷积
                对每个 user 取数据集中所有的数据, 而不是sample出其中一个, 作为正样本
            """
            p_items = np.nonzero(self.R[u, :])[0].tolist()
            # p_item = random.sample(p_items, 1)[0]
            # R_missing[u, p_item] = 0
            pos_items += p_items
            node_1 += [u] * len(p_items)

        adj_missing = self.adj_mat(R=R_missing)
        node_2 = [self.n_users + p for p in pos_items]
        all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.adj_miss: adj_missing})

        negative_items = []
        for u in users:
            """
                这里对 all_items (而不是neg_items) 进行 softmax 算 relevance_probability
                这样 generator 可以生成真实的样本
                例如 数据集中有 (u_0, i_3)， generator 也可以生成 (u_0, i_3)
            """
            # neg_items = list(set(range(self.n_items)) - set(np.nonzero(self.R[u, :])[0].tolist()))
            all_items = range(self.n_items)
            relevance_probability = all_score[u, all_items]
            relevance_probability = utils.softmax(relevance_probability)
            # 对u, 数据集中有多少个正样本，就在 relevance_probability 中 sample 多少个负样本
            p_items = np.nonzero(self.R[u, :])[0].tolist()
            neg_item = np.random.choice(all_items, size=len(p_items), p=relevance_probability).tolist()
            negative_items += neg_item

        node_2 += [self.n_users + p for p in negative_items]
        node_1 = node_1 * 2

        batch_size = len(node_1)
        assert batch_size == len(node_2)

        labels = [1.0]*(batch_size//2) + [0.0] * (batch_size//2)

        return adj_missing, node_1, node_2, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        users = random.sample(range(self.n_users), config.missing_edge)
        R_missing = np.copy(self.R)
        adj_missing = self.adj_mat(R=R_missing)
        node_1 = []
        node_2 = []

        for u in users:
            # 算 u 和所有 item 之间的 embedding 乘积
            i = list(range(self.n_users, self.n_node))
            relevance_probability = self.sess.run(self.generator.prob,
                                  feed_dict={
                                      self.generator.adj_miss: np.array(adj_missing),
                                      self.generator.u: np.array(u),
                                      self.generator.i: np.array(i)
                                  })
            pos_items = np.nonzero(self.R[u, :])[0].tolist()
            # 对u, 采 2 * len(pos_items) 个 样本, 这样对于 u 来说训练 D 和 G 的时候样本数量是一致的
            neg_item = np.random.choice(np.arange(self.n_items), 2*len(pos_items), p=relevance_probability)
            neg_item += data.n_users
            node_1 += 2 * len(pos_items) * [u]
            node_2 += neg_item.tolist()

        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.adj_miss: adj_missing,
                                          self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return adj_missing, node_1, node_2, reward

    def adj_mat(self, R, self_connection=True):
        # convert user-item graph to a big graph
        A = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items], dtype=np.float32)
        A[:self.n_users, self.n_users:] = R
        A[self.n_users:, :self.n_users] = R.T
        if self_connection:
            return np.identity(self.n_users + self.n_items, dtype=np.float32) + A
        return A


if __name__ == "__main__":
    spectral_gan = SpectralGAN(n_users=data.n_users, n_items=data.n_items, R=data.R)
    spectral_gan.train()
