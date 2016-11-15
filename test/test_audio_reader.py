"""Unit tests for audio reader."""
import threading
import time

import numpy as np
import tensorflow as tf

from wavenet import AudioReader

class TestAudioReaderClass(AudioReader):
    
    def __init__(self, validation, queue_size, sample_size, 
        n_train_samples, n_validation_samples):
        super(TestAudioReaderClass, self).__init__(validation, queue_size)
        self.sample_size = sample_size
        self.n_train_samples = n_train_samples
        self.n_validation_samples = n_validation_samples
        
    def get_audio_iterator(self, train):
        samples = [range(1, self.n_train_samples+1), range(-self.n_validation_samples, 0)]
        idx = 0 if train else 1
        for s in samples[idx]:
            audio = np.ones((self.sample_size, 1)) * s
            yield audio.astype(np.float32), str(s)
                

class TestAudioReader(tf.test.TestCase):

    def doTrainTest(self, queue_size, sample_size, n_train_samples, 
        n_validation_samples, batch_size, steps):
        reader = TestAudioReaderClass(False, queue_size, sample_size, 
            n_train_samples, n_validation_samples)
        batch = reader.dequeue(batch_size)
        with self.test_session() as sess:
            reader.start_threads(sess)
            try:
                next_value = 1
                for step in range(steps):
                    result = sess.run(batch)
                    self.assertEqual(batch_size, len(result))
                    self.assertEqual((sample_size, 1), result[0].shape)
                    for b in range(batch_size):
                        value = result[b][0, 0]
                        self.assertEqual(next_value, int(value))
                        next_value = 1 if value == n_train_samples else int(value)+1
            finally:
                reader.stop_threads(sess)
                
    def testTrain(self):
        self.doTrainTest(queue_size=4, sample_size=10, 
            n_train_samples=11, n_validation_samples=5, 
            batch_size=3, steps=9)
            
    def doValidationTest(self, queue_size, sample_size, n_train_samples, 
        n_validation_samples, batch_size, train_steps, val_steps):
        reader = TestAudioReaderClass(True, queue_size, sample_size, 
            n_train_samples, n_validation_samples)
        batch = reader.dequeue(batch_size)
        
        with self.test_session(use_gpu=True) as sess:
            reader.start_threads(sess)
            try:
                next_train_value = 1
                for step in range(train_steps):
                    result = sess.run(batch, {reader.train_flag: True})
                    self.assertEqual(batch_size, len(result))
                    self.assertEqual((sample_size, 1), result[0].shape)
                    for b in range(batch_size):
                        train_value = result[b][0, 0]
                        self.assertEqual(next_train_value, int(train_value))
                        next_train_value = 1 if train_value == n_train_samples else int(train_value)+1
                    # validation
                    next_val_value = -n_validation_samples
                    for val_step in range(val_steps):
                        val_result = sess.run(batch, {reader.train_flag: False})
                        self.assertEqual((sample_size, 1), val_result[0].shape)
                        self.assertEqual(batch_size, len(val_result))
                        for b in range(batch_size):
                            val_value = val_result[b][0, 0]
                            self.assertEqual(next_val_value, int(val_value))
                            next_val_value = -n_validation_samples if val_value == -1 else int(val_value)+1  
                    drain = reader.reset_queue(sess, False)
            finally:
                reader.stop_threads(sess) 
            
    def testValidation(self):
        for _ in range(1):
            self.doValidationTest(queue_size=4, sample_size=10, 
                n_train_samples=11, n_validation_samples=7, 
                batch_size=3, train_steps=9, val_steps=2)
      
if __name__ == '__main__':
    tf.test.main()
