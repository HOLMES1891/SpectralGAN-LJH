import tensorflow as tf
import config

class Discriminator(object):
    def __init__(self, data, n_node, n_layer):
        self.n_node = n_node
        self.data = data

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.Variable(tf.random_normal([self.n_node, config.emb_dim],
                                                                 mean=0.0, stddev=0.02, dtype=tf.float32),
                                                                 name='features')

            self.weight_matrix = tf.Variable(tf.random_normal([n_layer, config.emb_dim, config.emb_dim],
                                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                                               name='weight')

        self.adj_miss = tf.placeholder(tf.int32, shape=[n_node, n_node])
        self.node_id = tf.placeholder(tf.int32)
        self.node_neighbor_id = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        adj_miss = tf.cast(self.adj_miss, tf.float32)
        degree = tf.diag(tf.reciprocal(tf.reduce_sum(adj_miss, axis=1)))
        for l in range(n_layer):
            weight_for_l = tf.gather(self.weight_matrix, l)
            self.embedding_matrix = tf.nn.sigmoid(tf.matmul(tf.matmul(tf.matmul(degree, adj_miss),
                                                                      self.embedding_matrix),
                                                            weight_for_l))

        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                              self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1)

        self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) \
                + config.lambda_dis * ( tf.nn.l2_loss(self.node_neighbor_embedding) +
                                                tf.nn.l2_loss(self.node_embedding))

        user_embeddings, item_embeddings = tf.split(self.embedding_matrix, [self.data.n_users, self.data.n_items])
        self.all_score = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(tf.reduce_mean(self.loss))
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
