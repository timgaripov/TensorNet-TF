import numpy as np
import tensorflow as tf


def get_train_data(data_dir):
    f = np.load(data_dir + '/cifar.npz')
    images = np.reshape(f['train_images'], [-1, 32, 32, 3])
    labels = f['train_labels']
    return images, labels

def get_validation_data(data_dir):
    f = np.load(data_dir + '/cifar.npz')
    images = np.reshape(f['validation_images'], [-1, 32, 32, 3])
    labels = f['validation_labels']
    return images, labels
 

def get_input_data(FLAGS):
    t_images, t_labels = get_train_data(FLAGS.data_dir)    
    t_cnt = t_images.shape[0]
    train_images_ph = tf.placeholder(dtype=tf.float32, shape=[t_cnt, 32, 32, 3], name='train_images_ph')
    train_labels_ph = tf.placeholder(dtype=tf.int32, shape=[t_cnt], name='train_labels_ph')
    train_images = tf.Variable(train_images_ph, trainable=False, collections=[], name='train_images')
    train_labels = tf.Variable(train_labels_ph, trainable=False, collections=[], name='train_labels')

    train_image_input, train_label_input = tf.train.slice_input_producer([train_images, train_labels],
                                                                         shuffle=True,
                                                                         capacity=FLAGS.num_gpus * FLAGS.batch_size + 20,
                                                                         name='train_input')



    v_images, v_labels = get_validation_data(FLAGS.data_dir)
    v_cnt = v_images.shape[0]
    validation_images_ph = tf.placeholder(dtype=tf.float32, shape=[v_cnt, 32, 32, 3], name='validation_images_ph')
    validation_labels_ph = tf.placeholder(dtype=tf.int32, shape=[v_cnt], name='validation_labels_ph')
    validation_images = tf.Variable(validation_images_ph, trainable=False, collections=[], name='validation_images')
    validation_labels = tf.Variable(validation_labels_ph, trainable=False, collections=[], name='validation_labels')

    validation_image_input, validation_label_input = tf.train.slice_input_producer([validation_images, validation_labels],
                                                                                   shuffle=False,
                                                                                   capacity=FLAGS.batch_size + 20,
                                                                                   name='validation_input')

    result = {}
    result['train'] = {
        'images': t_images,
        'labels': t_labels,
        'image_input': train_image_input,
        'label_input': train_label_input
    }
    result['validation'] = {
        'images': v_images,
        'labels': v_labels,
        'image_input': validation_image_input,
        'label_input': validation_label_input
    }
    result['initializer'] = [
        train_images.initializer,
        train_labels.initializer,
        validation_images.initializer,
        validation_labels.initializer
    ]

    result['init_feed'] = {
        train_images_ph: t_images,
        train_labels_ph: t_labels,
        validation_images_ph: v_images,
        validation_labels_ph: v_labels
    }
    
    result['aux'] = {
        'mean': np.mean(t_images, axis=0),
        'std': np.std(t_images, axis=0)
    }


    return result
