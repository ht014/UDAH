
import tensorflow as tf
import numpy as np
import math
import sys
import prettytensor as pt
from deconv import deconv2d
sys.path.append('optimizers')
FLAGS = tf.app.flags.FLAGS

def protoloss(sc, tc):
    return tf.reduce_mean((tf.abs(sc - tc)))


class LeNetModel(object):

    def __init__(self, num_classes=1000, is_training=True, image_size=32, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.default_image_size = image_size
        self.is_training = is_training
        self.num_channels = 1
        self.mean = None
        self.bgr = False
        self.k = 2
        self.range = None
        self.featurelen = 10
        self.source_moving_centroid = tf.get_variable(name='source_moving_centroid',
                                                      shape=[num_classes, self.featurelen],
                                                      initializer=tf.zeros_initializer(), trainable=False)
        self.target_moving_centroid = tf.get_variable(name='target_moving_centroid',
                                                      shape=[num_classes, self.featurelen],
                                                      initializer=tf.zeros_initializer(), trainable=False)

        tf.summary.histogram('source_moving_centroid', self.source_moving_centroid)
        tf.summary.histogram('target_moving_centroid', self.target_moving_centroid)

    def inference(self, x, training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 5, 5, 20, 1, 1, padding='VALID', bn=True, name='conv1')
        pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(pool1, 5, 5, 50, 1, 1, padding='VALID', bn=True, name='conv2')
        pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name='pool2')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.contrib.layers.flatten(pool2)
        self.flattened = flattened
        fc1 = fc(flattened, 1250, 512, bn=False, name='fc1')
        fc2 = fc(fc1, 512, 10, relu=False, bn=False, name='fc2')
        self.fc1 = fc1
        self.fc2 = fc2
        self.score = fc2
        self.output = tf.nn.softmax(self.score)
        self.feature = fc2
        return self.score

    def adoptimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if   v.name.split('/')[0] in ['dt','ds']]
        D_weights = [v for v in var_list if 'weights' in v.name]
        D_biases = [v for v in var_list if 'biases' in v.name]
        print '=================Discriminator_weights====================='
        print D_weights
        print '=================Discriminator_biases====================='
        print D_biases

        self.Dregloss = 5e-4 * tf.reduce_mean([tf.nn.l2_loss(v) for v in var_list if 'weights' in v.name])
        D_op1 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.D_loss + self.Dregloss, var_list=D_weights)
        D_op2 = tf.train.MomentumOptimizer(learning_rate * 2.0, 0.9).minimize(self.D_loss + self.Dregloss,
                                                                              var_list=D_biases)
        D_op = tf.group(D_op1, D_op2)
        return D_op

    def adloss(self, x, xt, y, yt):
        with tf.variable_scope('reuse_inference') as scope:
            scope.reuse_variables()
            self.inference(x, training=True)
            source_fc1 = self.fc1
            source_feature = self.feature
            scope.reuse_variables()
            self.inference(xt, training=True)
            target_fc1 = self.fc1
            target_feature = self.feature
            target_pred = self.output

        with tf.variable_scope('gs') as scope:
            fake_ss = source_transform(source_fc1)
            scope.reuse_variables()
            fake_ts = source_transform(target_fc1)
        with tf.variable_scope('gt') as scope:
            fake_tt = target_transform(target_fc1)
            scope.reuse_variables()
            fake_st = target_transform(source_fc1)
        with tf.variable_scope('dt')  as scope:
            logist_st = target_dis(fake_st)
            scope.reuse_variables()
            logist_t = target_dis(xt)
        with tf.variable_scope('ds') as scope:
            logist_ts = source_dis(fake_ts)
            scope.reuse_variables()
            logist_s = source_dis(x)

        self.reconst_ss = fake_ss
        self.reconst_st = fake_st

        self.concat_feature = tf.concat([source_feature, target_feature], 0)

        source_result = tf.argmax(y, 1)
        target_result = tf.argmax(target_pred, 1)


        ones = tf.ones_like(source_feature)
        current_source_count = tf.unsorted_segment_sum(ones, source_result, self.num_classes)
        current_target_count = tf.unsorted_segment_sum(ones, target_result, self.num_classes)

        current_positive_source_count = tf.maximum(current_source_count, tf.ones_like(current_source_count))
        current_positive_target_count = tf.maximum(current_target_count, tf.ones_like(current_target_count))

        current_source_centroid = tf.divide(
            tf.unsorted_segment_sum(data=source_feature, segment_ids=source_result, num_segments=self.num_classes),
            current_positive_source_count)
        current_target_centroid = tf.divide(
            tf.unsorted_segment_sum(data=target_feature, segment_ids=target_result, num_segments=self.num_classes),
            current_positive_target_count)
        self.current_target_centroid = current_target_centroid

        source_decay = tf.constant(.3)
        target_decay = tf.constant(.3)

        self.source_decay = source_decay
        self.target_decay = target_decay

        source_centroid = (source_decay) * current_source_centroid + (1. - source_decay) * self.source_moving_centroid
        target_centroid = (target_decay) * current_target_centroid + (1. - target_decay) * self.target_moving_centroid

        self.Entropyloss = tf.constant(0.)
        self.Semanticloss = protoloss(source_centroid, target_centroid)
        theta1 = theta2 = 0.25
       
        source_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logist_s,
                                                    labels=tf.one_hot(source_result, self.num_classes )))

        target_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logist_t,
                                                    labels=tf.one_hot(target_result, self.num_classes )))
     


        # training generator
        source_loss += theta1* tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logist_ts,
                                                    labels=tf.one_hot(target_result, self.num_classes + 1)
                                                    )
        )
        target_loss += theta2* tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logist_st,
                                                    labels=tf.one_hot(source_result, self.num_classes + 1)
                                                    )
        )
        self.G_loss = source_loss
        self.D_loss = target_loss
        return self.G_loss, self.D_loss, source_centroid, target_centroid

    def loss(self, batch_x, batch_y=None ,S, is_discrete=False):
        with tf.variable_scope('reuse_inference') as scope:
            y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))

        if  is_discrete ==False:
            F2_dis = tf.sqrt(tf.reduce_sum(self.fc1**2,1))
            self.hash_loss  = ( tf.reduce_mean(
                tf.square(tf.matmul(self.fc1, tf.transpose(self.fc1)) / F2_dis - S))
        else:
            z=self.fc1
            l = z.get_shape()[0]
            z = z.reshape(l/self.k,self.k)
            u=self._gumbel_softmax(z, 1, sampling=True)
            hash_codes = tf.cast(tf.argmax(u, axis=2), tf.int32)
            self.hash_loss  = ( tf.reduce_mean(
                tf.square(tf.matmul(u, tf.transpose(u)) - S))
            

        return self.loss+is_discrete*self.hash_loss
    def _gumbel_dist(self, shape, eps=1e-20):
       U = tf.random_uniform(shape,minval=0,maxval=1)
       return -tf.log(-tf.log(U + eps) + eps)
   
    def _sample_gumbel_vectors(self, logits, temperature):
       y = logits + self._gumbel_dist(tf.shape(logits))
       return tf.nn.softmax( y / temperature)

    def _gumbel_softmax(self, logits, temperature, sampling=True):
      
       if sampling:
           y = self._sample_gumbel_vectors(logits, temperature)
       else:
           k = tf.shape(logits)[-1]
           y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
           y = tf.stop_gradient(y_hard - y) + y
       return y

    def optimize(self, learning_rate, train_layers, global_step, source_centroid, target_centroid):

        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in ['reuse_inference','gs','gt']]

        new_weights = [v for v in var_list if 'weights' in v.name or 'gamma' in v.name]
        new_biases = [v for v in var_list if 'biases' in v.name or 'beta' in v.name]

    

        self.F_loss = self.loss  + global_step * self.Semanticloss + global_step * self.G_loss + self.hash_loss*0.1
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
        with tf.control_dependencies(update_ops):
            train_op3 = tf.train.MomentumOptimizer(learning_rate * 1.0, 0.9).minimize(self.F_loss, var_list=new_weights)
            train_op4 = tf.train.MomentumOptimizer(learning_rate * 2.0, 0.9).minimize(self.F_loss, var_list=new_biases)

        with tf.control_dependencies([train_op3,train_op4]):
            update_sc = self.source_moving_centroid.assign(source_centroid)
            update_tc = self.target_moving_centroid.assign(target_centroid)
        return tf.group(update_sc, update_tc)

    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load('bvlc_alexnet.npy', encoding='bytes').item()

        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope('reuse_inference/' + op_name, reuse=True):
                print '=============================OP_NAME  ========================================'
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
                        print op_name, var
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
                        print op_name, var
                        session.run(var.assign(data))


 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, bn=False, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / groups, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
        if bn == True:
            bias = tf.contrib.layers.batch_norm(bias, scale=True)
        relu = tf.nn.relu(bias, name=scope.name)
        return relu

 

def source_dis(x):
    
    fc1 = fc(x, 4*4*128, 500, bn=False, name='fc1')
    fc2 = fc(fc1, 500, 10, relu=False, bn=False, name='fc2')
    return fc2


def target_dis(x):
    
    fc1 = fc(x, 4*4*128, 500, bn=False, name='fc1')
    fc2 = fc(fc1, 500, 10, relu=False, bn=False, name='fc2')
    return fc2

def source_transform(Z):
        with tf.variable_scope('xx'):
            return (pt.wrap(Z).
                    fully_connected(4 * 4 * 128)
                    )

def target_transform(Z):
        with tf.variable_scope('xx2'):
            return (pt.wrap(Z).
                    fully_connected(4 * 4 * 128)
                    )

def fc(x, num_in, num_out, name, relu=True, bn=False, stddev=0.001):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[num_out]))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if bn == True:
            act = tf.contrib.layers.batch_norm(act, scale=True)
        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def outer(a, b):
    a = tf.reshape(a, [-1, a.get_shape()[-1], 1])
    b = tf.reshape(b, [-1, 1, b.get_shape()[-1]])
    c = a * b
    return tf.contrib.layers.flatten(c)


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
