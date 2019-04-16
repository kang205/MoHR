import tensorflow as tf


class MoHR():
    def save(self, sess):
        self.saver.save(sess, self.model_name)

    def predict(self, item_i, item_j, relation, relation_emb):
        item_i = tf.expand_dims(item_i, 1)
        item_j = tf.expand_dims(item_j, 1)

        relation = tf.concat(
            [relation * self.gamma, (1 - self.gamma) * tf.ones([tf.shape(relation)[0], 1])], 1
        )

        return tf.reduce_sum(relation * (-tf.reduce_sum(tf.square(item_i + relation_emb - item_j), 2)), 1)

    def decision(self, inp):
        [i, u] = inp
        real_bsize = tf.shape(u)[0]
        batch_u_emb = tf.expand_dims(tf.nn.embedding_lookup(self.u_embeddings, u), 1)
        batch_i_emb = tf.expand_dims(tf.nn.embedding_lookup(self.i_embeddings, i), 1)
        batch_r_emb = tf.concat(
            [tf.tile(tf.expand_dims(self.r0_embeddings, 0), [real_bsize, 1, 1]),
             tf.tile(tf.expand_dims(self.r_embeddings, 0), [real_bsize, 1, 1])], 1
        )
        batch_r_bias = tf.tile(self.r_biases, [real_bsize, 1])
        pred = batch_r_bias - tf.reduce_sum(tf.square(batch_u_emb + batch_i_emb - batch_r_emb), 2)
        pred = tf.nn.softmax(pred)

        return pred

    def __init__(self, usernum, itemnum, Relationships, args):
        self.Relationships = Relationships
        self.usernum = usernum
        self.itemnum = itemnum
        self.d_emb = args.latent_dimension
        self.relnum = len(Relationships)
        self.gamma = args.gamma

        d_emb = args.latent_dimension
        learning_rate = args.learning_rate
        self.model_name = 'MoHR_%s_%d_%g_%g_%g_%g_%g_%g.ckpt' % (
        args.dataset, d_emb, learning_rate, args.alpha, args.beta, args.norm, args.lambda_bias, args.gamma)

        # Learnable Variables
        self.u_embeddings = tf.Variable(tf.nn.l2_normalize(tf.random_normal([usernum, d_emb],
                                                                            stddev=1 / (d_emb ** 0.5),
                                                                            dtype=tf.float32), 1))
        self.i_embeddings = tf.Variable(tf.nn.l2_normalize(tf.random_normal([itemnum, d_emb],
                                                                            stddev=1 / (d_emb ** 0.5),
                                                                            dtype=tf.float32), 1))
        self.r_embeddings = tf.Variable(tf.nn.l2_normalize(tf.random_normal([self.relnum, d_emb],
                                                                            stddev=1 / (d_emb ** 0.5),
                                                                            dtype=tf.float32), 1))
        self.r0_embeddings = tf.Variable(tf.nn.l2_normalize(tf.random_normal([1, d_emb],
                                                                             stddev=1 / (d_emb ** 0.5),
                                                                             dtype=tf.float32), 1))
        self.r_biases = tf.Variable(tf.zeros([1, self.relnum + 1], dtype=tf.float32))

        self.i_biases = tf.Variable(tf.zeros([itemnum], dtype=tf.float32))

        # Sample Batch
        self.batch_u = tf.placeholder(tf.int32, [None])
        real_bsize = tf.shape(self.batch_u)[0]

        self.batch_i = tf.placeholder(tf.int32, [None])
        self.batch_ui_r = tf.placeholder(tf.int32, [None])
        self.batch_ui_rp = tf.placeholder(tf.int32, [None])
        self.batch_j = tf.placeholder(tf.int32, [None])
        self.batch_jp = tf.placeholder(tf.int32, [None])

        self.batch_lp_i = tf.placeholder(tf.int32, [None])
        self.batch_lp_j = tf.placeholder(tf.int32, [None])
        self.batch_lp_r = tf.placeholder(tf.int32, [None])
        self.batch_lp_jp = tf.placeholder(tf.int32, [None])

        self.batch_lp_i_emb = tf.nn.embedding_lookup(self.i_embeddings, self.batch_lp_i)
        self.batch_lp_j_emb = tf.nn.embedding_lookup(self.i_embeddings, self.batch_lp_j)
        self.batch_lp_r_emb = tf.nn.embedding_lookup(self.r_embeddings, self.batch_lp_r)
        self.batch_lp_jp_emb = tf.nn.embedding_lookup(self.i_embeddings, self.batch_lp_jp)

        self.batch_u_emb = tf.nn.embedding_lookup(self.u_embeddings, self.batch_u)
        self.batch_i_emb = tf.nn.embedding_lookup(self.i_embeddings, self.batch_i)
        self.batch_j_emb = tf.nn.embedding_lookup(self.i_embeddings, self.batch_j)
        self.batch_jp_emb = tf.nn.embedding_lookup(self.i_embeddings, self.batch_jp)
        self.batch_j_bias = tf.nn.embedding_lookup(self.i_biases, self.batch_j)
        self.batch_jp_bias = tf.nn.embedding_lookup(self.i_biases, self.batch_jp)
        self.batch_lp_j_bias = tf.nn.embedding_lookup(self.i_biases, self.batch_lp_j)
        self.batch_lp_jp_bias = tf.nn.embedding_lookup(self.i_biases, self.batch_lp_jp)
        self.batch_r0 = tf.tile(tf.expand_dims(self.r0_embeddings, 0), [real_bsize, 1, 1])

        relation_weight = self.decision([self.batch_i, self.batch_u])

        pos_distances = self.batch_j_bias + self.predict(self.batch_i_emb, self.batch_j_emb, relation_weight,
                                                         tf.concat([self.batch_r0,
                                                                    tf.tile(tf.expand_dims(self.r_embeddings, 0),
                                                                            [real_bsize, 1, 1]),
                                                                    tf.expand_dims(self.batch_u_emb, 1)], 1))
        neg_distances = self.batch_jp_bias + self.predict(self.batch_i_emb, self.batch_jp_emb, relation_weight,
                                                          tf.concat([self.batch_r0,
                                                                     tf.tile(tf.expand_dims(self.r_embeddings, 0),
                                                                             [real_bsize, 1, 1]),
                                                                     tf.expand_dims(self.batch_u_emb, 1)], 1))

        lp_pos_distances = self.batch_lp_j_bias - tf.reduce_sum(
            tf.square(self.batch_lp_i_emb + self.batch_lp_r_emb - self.batch_lp_j_emb), 1)

        lp_neg_distances = self.batch_lp_jp_bias - tf.reduce_sum(
            tf.square(self.batch_lp_i_emb + self.batch_lp_r_emb - self.batch_lp_jp_emb), 1)

        self.lp_loss = -tf.reduce_mean(tf.log_sigmoid(lp_pos_distances - lp_neg_distances))

        o_uir = tf.reduce_sum(tf.one_hot(self.batch_ui_r, self.relnum + 1) * relation_weight, 1)
        o_uirp = tf.reduce_sum(tf.one_hot(self.batch_ui_rp, self.relnum + 1) * relation_weight, 1)
        self.rp_loss = -tf.reduce_mean(tf.log_sigmoid(o_uir - o_uirp))

        self.loss = -tf.reduce_mean(
            tf.log_sigmoid(pos_distances - neg_distances)) + args.alpha * self.lp_loss + args.beta * self.rp_loss

        self.loss += args.lambda_bias * sum(map(tf.nn.l2_loss,
                                                [self.batch_j_bias, self.batch_jp_bias, self.batch_lp_j_bias,
                                                 self.batch_lp_jp_bias, self.r_biases]))

        self.auc = tf.reduce_mean((tf.sign(-neg_distances + pos_distances) + 1) / 2)
        self.link_auc = tf.reduce_mean((tf.sign(-lp_neg_distances + lp_pos_distances) + 1) / 2)
        self.rp_auc = tf.reduce_mean((tf.sign(o_uir - o_uirp) + 1) / 2)

        self.gds = []
        self.gds.append(tf.train.AdamOptimizer(learning_rate).minimize(self.loss))
        with tf.control_dependencies(self.gds):
            self.gds.append(tf.assign(self.u_embeddings, tf.clip_by_norm(self.u_embeddings, args.norm, axes=[1])))
            self.gds.append(tf.assign(self.i_embeddings, tf.clip_by_norm(self.i_embeddings, args.norm, axes=[1])))
            self.gds.append(tf.assign(self.r_embeddings, tf.clip_by_norm(self.r_embeddings, args.norm, axes=[1])))
            self.gds.append(tf.assign(self.r0_embeddings, tf.clip_by_norm(self.r0_embeddings, args.norm, axes=[1])))

        self.saver = tf.train.Saver()
