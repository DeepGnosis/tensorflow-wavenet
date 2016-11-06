import fnmatch
import os
import re
import threading

import numpy as np
import tensorflow as tf
import pandas as pd


def find_files(directory, pattern='*.pickle'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_prices(directory):
    '''Generator that yields prices timeseries from the directory.
    Currently works with 1-dim prices only.
    '''
    files = find_files(directory)
    for filename in files:
        prices, _ = pd.read_pickle(filename)
        prices = prices.reshape(-1, 1)
        yield prices, filename


class PriceReader(object):
    '''Generic background price reader that preprocesses price files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 price_dir,
                 coord,
                 # sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.price_dir = price_dir
        # self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(price_dir):
            raise ValueError("No price files found in '{}'.".format(price_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_prices(self.price_dir)
            for audio, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
