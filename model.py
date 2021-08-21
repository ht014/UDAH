import numpy as np
import tensorflow as tf
from utils import *
from predict import predict_cv
from distance import distance
import npdistance
class UDAH:
    def __init__(self, A, X, sim, L, z, K=1,  n_hidden=[128],
                 max_iter=400, T=0.85, verbose=True):

        self.sim = sim
        X = X.astype(np.float32)
        self.Z = z
        self.X = X
        self.feed_dict = None
        self.src_inputs = tf.sparse_placeholder(tf.float64)
        self.src_y = tf.placeholder(tf.int32,[None])
        self.tgt_inputs = tf.sparse_placeholder(tf.float64)
        self.N, self.D = X.shape
        self.L = L
        self.num_class = 8
        self.max_iter = max_iter
        self.verbose = verbose
        self.pos_margin = tf.placeholder(tf.float64,[None])
        self.neg_margin = tf.placeholder(tf.float64, [None])
        self.Sim = tf.placeholder(tf.float64,[None,None])
        if n_hidden is None:
            n_hidden = [128]
        self.n_hidden = n_hidden
        self.batch_size = 400
        self.T = T
        train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
            A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
            undirected=(A != A.T).nnz == 0)
        A_train = edges_to_sparse(train_ones, self.N)
        hops = get_hops(A_train, K)
        num_edges = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}


        self.val_edges = val_edges = np.row_stack((val_ones, val_zeros))
        self.val_input = np.row_stack((self.X[val_edges[:, 0], :], self.X[val_edges[:, 1], :]))
        self.val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1

        self.source_moving_centroid = tf.get_variable(name='source_moving_centroid',
                                                      shape=[self.num_class, self.n_hidden[0]],
                                                      initializer=tf.zeros_initializer(), trainable=False)
        self.target_moving_centroid = tf.get_variable(name='target_moving_centroid',
                                                      shape=[num_class, self.n_hidden[0]],
                                                      initializer=tf.zeros_initializer(), trainable=False)
        self.build_encoder()
        self.__dataset_generator(hops, num_edges)
        self.build_all_loss()


    def build_encoder(self,domain='src'):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden
        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W_encoder{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float64,
                                initializer=w_init())
            b = tf.get_variable(name='b_encoder{}'.format(i), shape=[sizes[i]], dtype=tf.float64, initializer=w_init())

            if i == 1:
                if domain =='src':
                    encoded = tf.sparse_tensor_dense_matmul(self.src_inputs, W) + b
                else:
                    encoded = tf.sparse_tensor_dense_matmul(self.tgt_inputs, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)
        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float64, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float64, initializer=w_init())
        mu = tf.matmul(encoded, W_mu) + b_mu
        return mu,self._gumbel_softmax(self.mu)


    def build_disc_src(self,src_embedings):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = self.n_hidden + 2*self.n_hidden

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W_src{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float64,
                                initializer=w_init())
            b = tf.get_variable(name='b_src{}'.format(i), shape=[sizes[i]], dtype=tf.float64, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(src_embedings, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.num_class], dtype=tf.float64, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.num_class], dtype=tf.float64, initializer=w_init())
        logist_scr = tf.matmul(encoded, W_mu) + b_mu
        # log_probs_src = tf.nn.log_softmax(self.logist_scr, axis=-1)

        return logist_scr

    def build_disc_tgt(self,tgt_embedings):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = self.n_hidden + 2*self.n_hidden

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W_tgt{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float64,initializer=w_init())
            b = tf.get_variable(name='b_tgt{}'.format(i), shape=[sizes[i]], dtype=tf.float64, initializer=w_init())
            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(tgt_embedings, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_mu = tf.get_variable(name='W_tgt', shape=[sizes[-1], self.num_class], dtype=tf.float64, initializer=w_init())
        b_mu = tf.get_variable(name='b_tgt', shape=[self.num_class], dtype=tf.float64, initializer=w_init())
        logist_tgt = tf.matmul(encoded, W_mu) + b_mu

        return logist_tgt

    def _gumbel_dist(self, shape, eps=1e-20):
        U = tf.random_uniform(shape,minval=0,maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _sample_gumbel_vectors(self, logits, temperature):
        y = logits + self._gumbel_dist(tf.shape(logits))
        return tf.nn.softmax(y / temperature)

    def _gumbel_softmax(self, logits, temperature=1., sampling=True):

        if sampling:
            y = self._sample_gumbel_vectors(logits, temperature)
        else:

            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def build_contrastive_loss(self,mu):
        hop_anchor = tf.gather(mu,tf.range(self.batch_size))

        hop_pos =tf.gather(mu,tf.range(self.batch_size,2*self.batch_size))
        hop_neg =  tf.gather(mu,tf.range(self.batch_size*2,3*self.batch_size))
        eng_pos = self.F2_distance(hop_anchor,hop_pos)
        eng_neg = self.F2_distance(hop_anchor,hop_neg)
        basic_loss = tf.maximum(eng_pos - eng_neg + 5, 0.0) #  (self.pos_margin-self.neg_margin)*5.
        contras_loss = tf.reduce_mean(basic_loss)
        return contras_loss

    def build_hash_loss(self,hash_codes):
        hash_sim = distance(hash_codes, x2=None, pair=True, dist_type="euclidean2")
        hash_loss = tf.reduce_mean(tf.square(hash_sim -self.Sim))
        return hash_loss

    def build_cross_entropy_loss(self,logist_scr,src_y,tgt_logts,psud_tgt,mask):

        src_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist_scr,labels= src_y))
        tgt_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tgt_logts,labels = psud_tgt)*mask)
        return src_cls_loss,tgt_cls_loss

    def build_kl_loss(self,log_probs_src,log_probs_tgt):
        p = tf.exp(log_probs_src)
        neg_ent = tf.reduce_sum(p * log_probs_tgt, axis=-1)
        neg_cross_ent = tf.reduce_sum(p * log_probs_tgt, axis=-1)
        kl_loss = neg_ent - neg_cross_ent
        return kl_loss

    def build_center_alignment_loss(self,y,target_pred,mask,source_feature,target_feature):

        source_result = tf.argmax(y, 1)
        target_result = tf.argmax(target_pred, 1)

        ones = tf.ones_like(source_feature)
        current_source_count = tf.unsorted_segment_sum(ones, source_result, self.num_class)
        current_target_count = tf.unsorted_segment_sum(ones*mask, target_result*mask, self.num_class)

        current_positive_source_count = tf.maximum(current_source_count, tf.ones_like(current_source_count))
        current_positive_target_count = tf.maximum(current_target_count, tf.ones_like(current_target_count))
        current_source_centroid = tf.divide(
            tf.unsorted_segment_sum(data=source_feature, segment_ids=source_result, num_segments=self.num_classes),
            current_positive_source_count)
        current_target_centroid = tf.divide(
            tf.unsorted_segment_sum(data=target_feature*mask, segment_ids=target_result, num_segments=self.num_classes),
            current_positive_target_count)
        # self.current_target_centroid = current_target_centroid

        source_decay = tf.constant(.3)
        target_decay = tf.constant(.3)

        self.source_decay = source_decay
        self.target_decay = target_decay

        source_centroid = (source_decay) * current_source_centroid + (1. - source_decay) * self.source_moving_centroid
        target_centroid = (target_decay) * current_target_centroid + (1. - target_decay) * self.target_moving_centroid

        alignment_loss = tf.reduce_mean((tf.abs(source_centroid - target_centroid)))
        return alignment_loss,source_centroid ,target_centroid

    def inference(self):
        with tf.variable_scope('reuse_inference') as scope:
            src_emd, src_hash_codes = self.build_encoder(domain='src')
            scope.reuse_variables()
            tgt_emd, tgt_hash_codes = self.build_encoder(domain='tgt')

        src_emd_loss = self.build_contrastive_loss(src_emd)
        tgt_emd_loss = self.build_contrastive_loss(tgt_emd)
        src_hash_loss = self.build_hash_loss(src_hash_codes) # only for source domain.
        with tf.variable_scope('src_dis') as scope:
            logist_scr = self.build_disc_src(src_hash_codes)
            scope.reuse_variables()
            logist_tgt = self.build_disc_src(tgt_hash_codes)

        target_result = tf.argmax(log_probs_tgt, 1)
        ones = tf.ones_like(target_result)
        zeros = tf.zeros_like(target_result)
        logist_tgt_norm = tf.nn.log_softmax(logist_tgt, axis=-1)
        mask = tf.where(logist_tgt_norm[:,target_result] > self.T,ones,zeros)
        logist_tgt_ = self.build_disc_tgt(tgt_hash_codes)
        kl_loss = self.build_kl_loss(logist_tgt,logist_tgt_)

        src_cls_loss,tgt_cls_loss = self.build_cross_entropy_loss(logist_scr,self.src_y,logist_tgt_,target_result,mask)
        alignment_loss,source_centroid,target_centroid = self.build_center_alignment_loss(self.src_y,logist_tgt, mask,src_emd,tgt_emd)
        self.source_moving_centroid.assign(source_centroid)
        self.target_moving_centroid.assign(target_centroid)

        return src_emd_loss+tgt_emd_loss+0.005*src_hash_loss+kl_loss+src_cls_loss+tgt_cls_loss+alignment_loss*0.1

    def F2_distance(self, p1, p2):
        pos_dist = distance(p1, p2, pair=False, dist_type='euclidean2')
        return pos_dist

    def __dataset_generator(self, hops, scale_terms):
        def gen():
            while True:
                all_triplets, all_scale_terms =  to_triplets(sample_all_hops(hops), scale_terms)

                num = all_triplets.shape[0]
                if num >= self.batch_size:
                    its = int(num / self.batch_size)
                else:
                    its = 1
                    self.batch_size = num

                arr = np.arange(all_triplets.shape[0])

                np.random.shuffle(arr)
                np.random.shuffle(arr)
                for i in xrange(its):
                    range_index = arr[(i * self.batch_size):(i + 1) * self.batch_size]
                    triplet_batch = all_triplets[range_index]
                    scale_batch = all_scale_terms[range_index]
                    tp = []
                    # print self.X.sum(),self.X[triplet_batch[:,0],:].sum()
                    tp.append(self.X[triplet_batch[:, 0], :])
                    tp.append(self.X[triplet_batch[:, 1], :])
                    tp.append(self.X[triplet_batch[:, 2], :])
                    lb = []
                    lb.append(self.Z[triplet_batch[:, 0]])
                    lb.append(self.Z[triplet_batch[:, 1]])
                    lb.append(self.Z[triplet_batch[:, 2]])
                    lb = np.array(lb).reshape(-1)
                    
                    yield np.row_stack(tp).astype(np.float64),scale_batch,triplet_batch,\
                          lb.astype(np.float64)

        dataset = tf.data.Dataset.from_generator(gen, (tf.float64, tf.float64,tf.int32,tf.float64),
                                                 ([None, None], [None],[None,3],[None]))
        self.triplets, self.scale_terms , self.batch_index,self.labels= dataset.prefetch(1).make_one_shot_iterator().get_next()

    def __save_vars(self, sess):
        self.saved_vars = {var.name: (var, sess.run(var)) for var in tf.trainable_variables()}

    def __restore_vars(self, sess):
        for name in self.saved_vars:
                sess.run(tf.assign(self.saved_vars[name][0], self.saved_vars[name][1]))
    def construct_sim(self,label):
        a = np.ones((label.shape[0],label.shape[0]))*10.
        b = np.ones((label.shape[0],label.shape[0]))
        for i in xrange(label.shape[0]):
            for j in xrange(i,label.shape[0]):
                if label[i]==label[j] and label[i] >= 0:
                    a[i][j] = 0

                if label[i] < 0 or label[j] <0:
                    b[i][j] = 0
                    b[j][i] = 0
        return a,_

    def train(self,gpu_list='4'):
        self.all_loss = self.inference()
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.all_loss)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list,
                                                                          allow_growth=True)))
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.max_iter*20):
            batch,index_,labels = sess.run([self.triplets,self.batch_index,self.labels])
            _s,_ = self.construct_sim(labels)

            mu,loss, _= sess.run([self.mu,self.all_loss,train_op], {self.inputs: sparse_feeder(batch),
                                            self.pos_margin: self.sim[index_[:,0],index_[:,1]],
                                            self.neg_margin: self.sim[index_[:,0],index_[:,2]],
                                            self.Sim: _s,  self.src_y:labels               })
            if self.verbose and epoch % 50 == 0:
                print('epoch: {:3d}, loss: {:.4f}, val_auc: {:.4f}, val_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap))
        return sess
