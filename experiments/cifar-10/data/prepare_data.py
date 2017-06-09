################################################################
#           Load and unpack CIFAR-10 python version            #
# from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz #
#                                                              #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  #
#            Run this scipt with python2 only                  #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  #
################################################################

import pickle
import numpy as np

batches_dir = 'cifar-10-batches-py' 

def unpickle(fname):
    fo = open(fname, 'rb')
    d = pickle.load(fo)
    fo.close()
    data = np.reshape(d['data'], [-1, 32, 32, 3], order='F')
    data = np.transpose(data, [0, 2, 1, 3])
    data = np.reshape(data, [-1, 32*32*3])
    labels = np.array(d['labels'], dtype='int8')
    return data, labels

for x in range(1, 6):
    fname = batches_dir + '/data_batch_' + str(x)
    data, labels = unpickle(fname)
    if x == 1:
        train_images = data
        train_labels = labels
    else:
        train_images = np.vstack((train_images, data))
        train_labels = np.concatenate((train_labels, labels))

validation_images, validation_labels = unpickle(batches_dir + '/test_batch')

print(train_images.shape, validation_images.shape)
print(train_labels.shape, validation_labels.shape)
np.savez_compressed('cifar', train_images=train_images, validation_images=validation_images,
                    train_labels=train_labels, validation_labels=validation_labels)


    







