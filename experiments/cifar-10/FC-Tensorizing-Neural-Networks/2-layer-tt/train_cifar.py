from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import datetime
import shutil
import time
import tensorflow.python.platform
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import net

tf.set_random_seed(12345)
np.random.seed(12345)

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 40000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('overview_steps', 100, 'Overview period')
flags.DEFINE_integer('evaluation_steps', 1000, 'Overview period')
flags.DEFINE_string('data_dir', '../../data/', 'Directory to put the training data.')
flags.DEFINE_string('log_dir', './log', 'Directory to put log files.')

def fill_feed_dict(batch, train_phase=True):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        batch: Tuple (Images, labels)
        evaluation: boolean, used to set dropout_rate to 1 in case of evaluation
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    graph = tf.get_default_graph()
    images_ph = graph.get_tensor_by_name('placeholder/images:0')
    labels_ph = graph.get_tensor_by_name('placeholder/labels:0')
    train_phase_ph = graph.get_tensor_by_name('placeholder/train_phase:0')

    images_feed, labels_feed = batch
    feed_dict = {
        images_ph: images_feed,
        labels_ph: labels_feed,
        train_phase_ph: train_phase
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            loss,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    sum_loss = 0.0
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set.next_batch(FLAGS.batch_size),
                                   train_phase=False)
        res = sess.run([loss, eval_correct], feed_dict=feed_dict)
        sum_loss += res[0]
        true_count += res[1]
    precision = true_count / num_examples
    avg_loss = sum_loss / (num_examples / FLAGS.batch_size)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f  Loss: %.2f' %
          (num_examples, true_count, precision, avg_loss))
    return precision, avg_loss

def run_training(extra_opts={}):
    start = datetime.datetime.now()
    start_str = start.strftime('%d-%m-%Y_%H_%M')
    train, validation = input_data.read_data_sets(FLAGS.data_dir)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        net.build(extra_opts)
        #Precision summaries
        precision_train = tf.Variable(0.0, trainable=False, name='precision_train')
        precision_validation = tf.Variable(0.0, trainable=False, name='precision_validation')

        precision_train_summary = tf.summary.scalar('precision/train',
                                                    precision_train)

        precision_validation_summary = tf.summary.scalar('precision/validation',
                                                         precision_validation)
        graph = tf.get_default_graph()
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_tensor_by_name('train_op:0')
        correct_count = graph.get_tensor_by_name('correct_count:0')
        #Create summary stuff
        regular_summaries_names = ['loss', 'learning_rate']
        regular_summaries_list = []
        for name in regular_summaries_names:
            summary = graph.get_tensor_by_name('summary/' + name + ':0')
            regular_summaries_list.append(summary)
        regular_summaries = tf.summary.merge(regular_summaries_list, name='summary/regular_summaries')
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(tf.global_variables())
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(graph=graph,
                          config=tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads = 3))
        sess.run(init)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir,
                                               graph=tf.get_default_graph())
        # And then after everything is built, start the training loop.
        for step in xrange(1, FLAGS.max_steps + 1):
            start_time = time.time()
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(train.next_batch(FLAGS.batch_size))
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.


            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time
            # Write the summaries and print an overview fairly often.
            if step % FLAGS.overview_steps == 0:
                # Print status to stdout.
                data_per_sec = FLAGS.batch_size / duration
                print('Step %d: loss = %.2f (%.3f sec) [%.2f data/s]' %
                      (step, loss_value, duration, data_per_sec))
                # Update the events file.
                summary_str = sess.run(regular_summaries, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step) % FLAGS.evaluation_steps == 0 or step == FLAGS.max_steps:
                saver.save(sess, FLAGS.log_dir +'/checkpoint', global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                precision_t, obj_t = do_eval(sess,
                                             correct_count,
                                             loss,
                                             train)
                sess.run(precision_train.assign(precision_t))

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                precision_v, obj_v = do_eval(sess,
                                             correct_count,
                                             loss,
                                             validation)
                sess.run(precision_validation.assign(precision_v))

                summary_str_0, summary_str_1 = sess.run([precision_train_summary, precision_validation_summary])
                summary_writer.add_summary(summary_str_0, step)
                summary_writer.add_summary(summary_str_1, step)

                os.makedirs('./results', exist_ok=True)
                res_file = open('./results/res_' + str(start_str), 'w')

                res_file.write('Iterations: ' + str(step) + '\n')
                now = datetime.datetime.now()
                delta = now - start
                res_file.write('Learning time: {0:.2f} minutes\n'.format(delta.total_seconds() / 60.0))
                res_file.write('Train precision: {0:.5f}\n'.format(precision_t))
                res_file.write('Train loss: {0:.5f}\n'.format(obj_t))
                res_file.write('Validation precision: {0:.5f}\n'.format(precision_v))
                res_file.write('Validation loss: {0:.5f}\n'.format(obj_v))
                res_file.write('Extra opts: ' + str(extra_opts) + '\n')
                res_file.write('Code:\n')
                net_file = open('./net.py', 'r')
                shutil.copyfileobj(net_file, res_file)
                net_file.close()
                res_file.close()

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
