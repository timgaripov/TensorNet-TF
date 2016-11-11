import tensorflow as tf
import numpy as np

class DataSet(object):
    def __init__(self, images, labels):
        """Construct a DataSet.
        """
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % 
                                                    (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(data_dir):
    f = np.load(data_dir + '/cifar.npz')
    train_images = f['train_images'].astype('float32')
    train_labels = f['train_labels']

    validation_images = f['validation_images'].astype('float32')
    validation_labels = f['validation_labels']

    mean = np.mean(train_images, axis=0)[np.newaxis, :]
    std = np.std(train_images, axis=0)[np.newaxis, :]

    train_images = (train_images - mean) / std;
    validation_images = (validation_images - mean) / std;

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    return train, validation
