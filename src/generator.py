import tensorflow as tf
from src import config
from src import test
from src.SpectralGAN import data


class Generator(object):
    """
        这个generator 被设计成 每次只能处理 一个 user
        self.u = tf.placeholder(tf.int32) 不能传入一个列表
    """
    def __init__(self, n_node, n_layer):
        self.n_node = n_node

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.Variable(tf.random_normal([self.n_node, config.emb_dim],
                                                                 mean=0.01, stddev=0.02, dtype=tf.float32),
                                                                 name='features')

            self.weight_matrix = tf.Variable(tf.random_normal([n_layer, config.emb_dim, config.emb_dim],
                                                               mean=0.01, stddev=0.02, dtype=tf.float32),
                                                               name='weight')

        self.adj_miss = tf.placeholder(tf.int32, shape=[n_node, n_node])
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.u)    # batch_size * n_embed
        self.i_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.i)    # 计算正则项的时候用

        adj_miss = tf.cast(self.adj_miss, tf.float32)
        degree = tf.diag(tf.reciprocal(tf.reduce_sum(adj_miss, axis=1)))
        for l in range(n_layer):
            weight_for_l = tf.gather(self.weight_matrix, l)
            self.embedding_matrix = tf.nn.leaky_relu(tf.matmul(tf.matmul(tf.matmul(degree, adj_miss),
                                                        self.embedding_matrix),
                                                   weight_for_l))

        self.user_embeddings, self.item_embeddings = tf.split(self.embedding_matrix, [data.n_users, data.n_items])

        # 对某个 u 计算 所有 item 的 score
        self.score = tf.reduce_sum(self.u_embedding * self.item_embeddings, axis=1)
        # self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)
        self.prob = tf.clip_by_value(tf.nn.softmax(self.score), 1e-5, 1)  # 这里计算 prob 的时候用 softmax 而不是 sigmoid

        self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + config.lambda_gen * (
                    tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding))

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(tf.reduce_mean(self.loss))

        # ---------------------
        # 为 D 生成全体的分布
        # ---------------------
        self.all_score = tf.matmul(self.user_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True)

