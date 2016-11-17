import fnmatch
import os
import re
import threading
import random
from abc import ABCMeta, abstractmethod

# import librosa
import numpy as np
import tensorflow as tf
import pandas as pd


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(files, sample_rate):
    '''Generator that yields audio waveforms from the files.'''
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def load_generic_prices(files):
    '''Generator that yields prices timeseries from the directory.
    Currently works with 1-dim prices only.
    '''
    for filename in files:
        # print(filename)
        prices = pd.read_pickle(filename)
        prices = prices.values.reshape(-1, 1)
        yield prices, filename


def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        yield audio, speaker_id


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
            
class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_audio_iterator(self, train):
        yield None, ''
    
    def __init__(self,
                 validation=True,
                 queue_size=256):
        self.coord = tf.train.Coordinator()
        self.validation = validation
        self.threads = []
        # 0 is train, 1 is validation
        num_of_queues = 2 if self.validation else 1
        self.reset_locks = [[None]*2]*num_of_queues # 0 is start lock, 1 is end lock
        self.placeholder = [None]*num_of_queues
        self.queue = [None]*num_of_queues
        self.enqueue = [None]*num_of_queues
        for i in range(num_of_queues):
            self.placeholder[i] = tf.placeholder(dtype=tf.float32, shape=None, 
                name=('train_samples' if i==0 else 'validation_samples'))
            self.queue[i] = tf.PaddingFIFOQueue(queue_size,
                                             ['float32'],
                                             shapes=[(None, 1)])
            self.enqueue[i] = self.queue[i].enqueue([self.placeholder[i]])
        self.train_flag = tf.placeholder(tf.bool) if self.validation else None

    def dequeue(self, num_elements):
        if self.validation:
            q = tf.QueueBase.from_list(tf.cond(self.train_flag, 
                lambda: tf.constant(0), lambda: tf.constant(1)), 
                [self.queue[0], self.queue[1]])
            output = q.dequeue_many(num_elements)
        else:
            output = self.queue[0].dequeue_many(num_elements)
        return output

    def thread_main(self, sess, train):
        queue_index = 0 if train else 1
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = self.get_audio_iterator(train)
            for audio, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                try:
                    sess.run(self.enqueue[queue_index],
                             feed_dict={self.placeholder[queue_index]: audio})
                except tf.errors.CancelledError:
                    stop = True
                    break
                if self.reset_locks[queue_index][0] is not None:
                    self.reset_locks[queue_index][0].set() # unlock and allow reset_queue to start draining
                    self.reset_locks[queue_index][1].wait() # wait for draining to complete before continue this thread
                    self.reset_locks[queue_index] = [None, None] # reset locks
                    break           

    def start_threads(self, sess, n_threads=1):
        for i in range(n_threads):
            thread = threading.Thread(target=self.thread_main, 
                name='Train-'+str(i), args=(sess, True))
            thread.start()
            self.threads.append(thread)
        if self.validation:
            thread = threading.Thread(target=self.thread_main, 
                name='Validation', args=(sess, False))
            thread.start()
            self.threads.append(thread)            
        return self.threads

    def stop_threads(self, sess):
        for q in self.queue:
            sess.run([q.close(cancel_pending_enqueues=True)])
        self.coord.request_stop()
        self.coord.join(self.threads)  
        
    def reset_queue(self, sess, train): # Note this won't work for single queue/multiple threads
        queue_index = 0 if train else 1
        self.reset_locks[queue_index] = [threading.Event(), threading.Event()]
        self.reset_locks[queue_index][0].wait() # wait for queue thread to signal the start of drain
        drain = self.drain_queue(sess, self.queue[queue_index])
        self.reset_locks[queue_index][1].set() # signal the end of draining so that queue thread can continue
        return drain
    
    def drain_queue(self, sess, queue): # do not run this on queue thread, otherwise strange error may appear
        num_remaining = sess.run([queue.size()])
        if num_remaining == 0:
            return None
        return sess.run([queue.dequeue_many(num_remaining)])

            
class DirectoryAudioReader(AudioReader):
    def __init__(self,
                 audio_dir,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 validation=True,
                 validation_split=0.1,
                 queue_size=256): 
        super(DirectoryAudioReader, self).__init__(validation, queue_size)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        if validation: # split files
            files = find_files(audio_dir)
            random.shuffle(files)
            split = int(len(files)*(1 - validation_split))
            self.train_files = files[:split]
            self.validation_files = files[split:]
        else:
            self.train_files = find_files(audio_dir)
            self.validation_files = None

    def get_audio_iterator(self, train):
        buffer_ = np.array([])
        files = self.train_files if train else self.validation_files
        iterator = load_generic_audio(files, self.sample_rate)
        for audio, filename in iterator:
            if self.silence_threshold is not None:
                # Remove silence
                audio = trim_silence(audio[:, 0], self.silence_threshold)
                if audio.size == 0:
                    print("Warning: {} was ignored as it contains only "
                          "silence. Consider decreasing trim_silence "
                          "threshold, or adjust volume of the audio."
                          .format(filename))

            if self.sample_size:
                # Cut samples into fixed size pieces
                buffer_ = np.append(buffer_, audio)
                while len(buffer_) > self.sample_size:
                    piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                    yield piece, filename
                    buffer_ = buffer_[self.sample_size:]
            else:
                yield audio, filename


class DirectoryPriceReader(AudioReader):
    def __init__(self,
                 audio_dir,
                 sample_size=None,
                 validation=True,
                 val_sample_size=None,
                 queue_size=256):
        super(DirectoryPriceReader, self).__init__(validation, queue_size)
        self.audio_dir = audio_dir
        self.sample_size = sample_size
        self.val_sample_size = val_sample_size
        if validation: # split files
            self.train_files = find_files(audio_dir, pattern="*train.pickle")
            self.validation_files = find_files(audio_dir, pattern="*val.pickle")
        else:
            self.train_files = find_files(audio_dir, pattern="*.pickle")
            self.validation_files = None

    def get_audio_iterator(self, train):
        buffer_ = np.array([])
        files = self.train_files if train else self.validation_files
        sample_size = self.sample_size if train else self.val_sample_size
        iterator = load_generic_prices(files)
        for audio, filename in iterator:
            if sample_size:
                # Cut samples into fixed size pieces
                buffer_ = np.append(buffer_, audio)
                # print('train', train, 'sample_size', sample_size, 'audio size', audio.shape, 'buffer size', buffer_.shape)
                while len(buffer_) >= sample_size:
                    piece = np.reshape(buffer_[:sample_size], [-1, 1])
                    yield piece, filename
                    buffer_ = buffer_[sample_size:]
            else:
                yield audio, filename
