 import os, sys

GPU = (sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
import numpy as np
import tensorflow as tf
import datetime
from mstnmodel import LeNetModel
from mn import MNIST
from sv import SVHN
from preprocessing import preprocessing
import scipy.io
import math
from tensorflow.contrib.tensorboard.plugins import projector

tf.app.flags.DEFINE_float('learning_rate', 5e-3, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 100000, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1',
                           'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '256,257',
                           'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')

NUM_CLASSES = 10

TRAIN_FILE = 'svhn'
TEST_FILE = 'mnist'
print TRAIN_FILE + '  --------------------------------------->   ' + TEST_FILE
print TRAIN_FILE + '  --------------------------------------->   ' + TEST_FILE
print TRAIN_FILE + '  --------------------------------------->   ' + TEST_FILE

TRAIN = SVHN('data/svhn', split='train', shuffle=True)
VALID = MNIST('data/mnist', split='train', shuffle=True)
TEST = MNIST('data/mnist', split='test', shuffle=False)

FLAGS = tf.app.flags.FLAGS
MAX_STEP = 10000

def decay(start_rate, epoch, num_epochs):
    return start_rate / pow(1 + 0.001 * epoch, 0.75)

def adaptation_factor(x):
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return min(lamb, 1.0)
def create_S(label):
    length = len(label)
    s = np.ones((length,length)) * -1.
    for i in xrange(length):
        for j in xrange(length):
            if label[i] == label[j]:
                s[i][j] = 1
    return s

def merge_images(sources, targets, tar2):
    _,  h, w, _= sources.shape
    row = int(np.sqrt(FLAGS.batch_size))
    merged = np.zeros([ (row + 1) * h, (row + 1) * w * 3,3])
    for idx, (s, t, t2) in enumerate(zip(sources, targets, tar2)):
        i = idx // row
        j = idx % row
        merged[  i * h:(i + 1) * h, (j * 3) * h:(j * 3 + 1) * h ,:] = s
        merged[  i * h:(i + 1) * h, (j * 3 + 1) * h:(j * 3 + 2) * h,:] = t
        merged[  i * h:(i + 1) * h, (j * 3 + 2) * h:(j * 3 + 3) * h,:] = t2
    return merged

def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    adlamb = tf.placeholder(tf.float32, name='adlamb')
    num_update = tf.placeholder(tf.float32, name='num_update')
    decay_learning_rate = tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    time = tf.placeholder(tf.float32, [1])

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = LeNetModel(num_classes=NUM_CLASSES, image_size=32, is_training=is_training,
                       dropout_keep_prob=dropout_keep_prob)
    # Placeholders
    x_s = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
    x_t = tf.placeholder(tf.float32, [None, 32, 32, 1], name='xt')
    Sx = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.batch_size], name='S')
    x = preprocessing(x_s, model)

    xt = preprocessing(x_t, model)
    tf.summary.image('Source Images', x)
    tf.summary.image('Target Images', xt)
    print 'x_s ', x_s.get_shape()
    print 'x ', x.get_shape()
    print 'x_t ', x_t.get_shape()
    print 'xt ', xt.get_shape()
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='yt')
    loss = model.loss(x, y,S=Sx)
    # Training accuracy of the model
    source_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    source_correct = tf.reduce_sum(tf.cast(source_correct_pred, tf.float32))
    source_accuracy = tf.reduce_mean(tf.cast(source_correct_pred, tf.float32))

    G_loss, D_loss, sc, tc = model.adloss(x, xt, y, yt)

    # Testing accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    update_op = model.optimize(decay_learning_rate, train_layers, adlamb, sc, tc)

    D_op = model.adoptimize(decay_learning_rate, train_layers)
    optimizer = tf.group(update_op, D_op)

    train_writer = tf.summary.FileWriter('./log/tensorboard')
    train_writer.add_graph(tf.get_default_graph())
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = model.feature.name
    embedding.metadata_path = 'domain.csv'
    projector.visualize_embeddings(train_writer, config)
    tf.summary.scalar('G_loss', model.G_loss)
    tf.summary.scalar('D_loss', model.D_loss)
    tf.summary.scalar('C_loss', model.loss)
    tf.summary.scalar('SA_loss', model.Semanticloss)
    tf.summary.scalar('Training Accuracy', source_accuracy)
    tf.summary.scalar('Testing Accuracy', accuracy)
    merged = tf.summary.merge_all()

    print '============================GLOBAL TRAINABLE VARIABLES ============================'
    print tf.trainable_variables()
    # print '============================GLOBAL VARIABLES ======================================'
    # print tf.global_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # saver.restore(sess,'log/checkpoint')
        # Load the pretrained weights
        # model.load_original_weights(sess, skip_layers=train_layers)
        train_writer.add_graph(sess.graph)
        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        # print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))
        gd = 0
        step = 1
        for epoch in range(300000):
            # Start training
            gd += 1
            lamb = adaptation_factor(gd * 1.0 / MAX_STEP)
            # rate=decay(FLAGS.learning_rate,gd,MAX_STEP)
            power = gd / 10000
            rate = FLAGS.learning_rate
            tt = pow(0.1, power)
            batch_xs, batch_ys = TRAIN.next_batch(FLAGS.batch_size)
            SS = create_S(np.argmax(batch_ys, 1))
            Tbatch_xs, Tbatch_ys = VALID.next_batch(FLAGS.batch_size)
            summary, _, closs, gloss, dloss, smloss,l1_loss,fake_ss2,fake_st2 , hloss= sess.run(
                [merged, optimizer, model.loss, model.G_loss, model.D_loss, model.Semanticloss,model.l1_loss,model.reconst_ss,model.reconst_st,model.hash_loss],
                feed_dict={x_s: batch_xs, x_t: Tbatch_xs, time: [1.0 * gd], decay_learning_rate: rate, adlamb: lamb,
                           is_training: True, y: batch_ys, dropout_keep_prob: 0.5, yt: Tbatch_ys,Sx:SS})
            train_writer.add_summary(summary, gd)
             
            step += 1
            if gd % 250 == 0:
                path = os.path.join('./samples', 'sample-%d-s-m.png' % (step + 1))
                merged2 =merge_images(batch_xs,fake_ss2,fake_st2)
                scipy.misc.imsave(path, merged2)
                epoch = gd / (72357 / 100)
                print 'lambda: ', lamb
                print 'rate: ', rate
                print 'Epoch {5:<10} Step {3:<10} C_loss {0:<10} G_loss {1:<10} D_loss {2:<10} Sem_loss {4:<10}, l1_loss {6:<10}'.format(
                    closs, gloss, dloss, gd, smloss, epoch,l1_loss)
             
if __name__ == '__main__':
    tf.app.run()
