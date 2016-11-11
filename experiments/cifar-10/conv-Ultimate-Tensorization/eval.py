from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import datetime
import shutil
import imp
import time
import tensorflow.python.platform
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import re
import input_data
import sys

import shutil

net = None

tf.set_random_seed(12345)
np.random.seed(12345)

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('net_module', None, 'Module with architecture description.')
flags.DEFINE_string('log_dir', None, 'Directory with log files.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('data_dir', '../data/', 'Directory to put the training data.')

flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

def tower_loss_and_eval(images, labels, train_phase, cpu_variables=False):
    logits = net.inference(images, train_phase, cpu_variables=cpu_variables)
    losses = net.losses(logits, labels)
    total_loss = tf.add_n(losses, name='total_loss')
    evaluation = net.evaluation(logits, labels)
    return total_loss, evaluation

def evaluate(sess,
             loss,
             evaluation,
             train_or_val,
             images_ph,
             images, 
             labels_ph,
             labels):
    fmt_str = 'Evaluation [%s]. Batch %d/%d (%d%%). Speed = %.2f sec/b, %.2f img/sec. Batch_loss = %.2f. Batch_precision = %.2f' 

    num_batches = labels.size // FLAGS.batch_size
    assert labels.size % FLAGS.batch_size == 0, 'Batch size must divide evenly into the dataset sizes.'
    assert images.shape[0] == labels.size, 'Images count must be equal to labels count'

    sum_loss = 0.0
    sum_correct = 0.0
    
    w = os.get_terminal_size().columns
    sys.stdout.write(('=' * w + '\n') * 2)
    sys.stdout.write('\n')
    sys.stdout.write('Evaluation [%s]' % train_or_val)

    cum_t = 0.0
    for bid in range(num_batches):
        b_images = images[bid * FLAGS.batch_size:(bid + 1) * FLAGS.batch_size]
        b_labels = labels[bid * FLAGS.batch_size:(bid + 1) * FLAGS.batch_size]
        start_time = time.time()
        loss_val, eval_val = sess.run([loss, evaluation], feed_dict={images_ph: b_images, labels_ph: b_labels})
        duration = time.time() - start_time

        cum_t += duration
        sec_per_batch = duration
        img_per_sec = FLAGS.batch_size / duration

        
        sum_loss += loss_val * FLAGS.batch_size
        sum_correct += np.sum(eval_val)
        
        if cum_t > 0.5:
            sys.stdout.write('\r' + fmt_str % (
                train_or_val,
                bid + 1,
                num_batches,
                int((bid + 1) * 100.0 / num_batches),
                sec_per_batch,
                img_per_sec,
                loss_val,
                np.mean(eval_val) * 100.0
            ))
            sys.stdout.flush()
            cum_t = 0.0
    
    sys.stdout.write(('\r' + fmt_str + '\n') % (
        train_or_val,
        num_batches,
        num_batches,
        int(100.0),
        sec_per_batch,
        img_per_sec,
        loss_val,
        np.mean(eval_val) * 100.0
    ))

    sys.stdout.write('%s loss = %.2f. %s precision = %.2f.\n\n' % (
        train_or_val,
        sum_loss / labels.size,
        train_or_val,
        sum_correct / labels.size * 100.0
    ))
           
def run_eval(chkpt):
    global net
    net = imp.load_source('net', FLAGS.net_module)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        train_phase = tf.constant(False, name='train_phase', dtype=tf.bool)
        
        t_images, t_labels = input_data.get_train_data(FLAGS.data_dir)
        aux = {
            'mean': np.mean(t_images, axis=0),
            'std': np.std(t_images, axis=0)
        }
        v_images, v_labels = input_data.get_validation_data(FLAGS.data_dir)  

        images_ph = tf.placeholder(tf.float32, shape=[None] + list(t_images.shape[1:]), name='images_ph')
        labels_ph = tf.placeholder(tf.int32, shape=[None], name='labels_ph')

        images = net.aug_eval(images_ph, aux)
        with tf.device('/gpu:0'):
            with tf.name_scope('tower_0') as scope:
                loss, evaluation = tower_loss_and_eval(images, labels_ph, train_phase)

        
        variable_averages =  tf.train.ExponentialMovingAverage(0.999)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        saver = tf.train.Saver(tf.all_variables())
        ema_saver = tf.train.Saver(variable_averages.variables_to_restore())

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        

                
        saver.restore(sess, chkpt)
        ema_saver.restore(sess, chkpt)
        sys.stdout.write('Checkpoint "%s" restored.\n' % (chkpt))
        evaluate(sess, loss, evaluation, 'Train', images_ph, t_images, labels_ph, t_labels)
        evaluate(sess, loss, evaluation, 'Validation', images_ph, v_images, labels_ph, v_labels)
            
def main(_):
    latest_chkpt = tf.train.latest_checkpoint(FLAGS.log_dir)
    if latest_chkpt is not None:
        sys.stdout.write('Checkpoint "%s" found.\n' % latest_chkpt)
        run_eval(latest_chkpt)
    else:
        sys.stdout.write('Checkpoint not found.\n')

if __name__ == '__main__':
    tf.app.run()
