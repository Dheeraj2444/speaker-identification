# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from model_lstm import AudioEventDetector
from audioset import vggish_input, vggish_params, vggish_postprocess, vggish_slim
# aed = AudioEventDetector()

import speech_recognition as sr
import os
import time
import config

cfg = config.Config()
class_mapping = {k:v for k,v in zip(range(cfg.NUM_CLASSES), cfg.CLASS_INDS)}

klass AudioRec(object):

    def __init__(self):
        self.r = sr.Recognizer()
        self.src = sr.Microphone()
        with self.src as source:
            print("Calibrating microphone...")
            self.r.adjust_for_ambient_noise(source, duration=2)


    def listen(self, save_path):
        with self.src as source:
            print("Recording ...")
            # record for a maximum of 10s
            audio = self.r.listen(source, phrase_time_limit=10)
        # write audio to a WAV file
        with open(save_path, "wb") as f:
            f.write(audio.get_wav_data())


class VGGish(object):
    # Paths to downloaded VGGish files.
    CHECKPOINT_PATH = 'audioset/vggish_model.ckpt'
    PCA_PARAMS_PATH = 'audioset/vggish_pca_params.npz'

    def __init__(self):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(self.sess, self.CHECKPOINT_PATH)


    def extract_features(self, wav_path):
        # Produce a batch of log mel spectrogram examples.
        input_batch = vggish_input.wavfile_to_examples(wav_path)
        if input_batch.shape[0] < 1:
            print('{}: Audio sample shorter than 1 second. Ignoring ...', os.path.basename(wav_path))
            return None

        features_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = self.sess.run([embedding_tensor],
                                    feed_dict={features_tensor: input_batch})
        # Postprocess the results to produce whitened quantized embeddings.
        pproc = vggish_postprocess.Postprocessor(self.PCA_PARAMS_PATH)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        return (np.float32(postprocessed_batch) - 128.) / 128.


    def __del__(self):
        print ("Closing the session..")
        self.sess.close()


def pad_sequences(mini_batch):
    """Function to pad 0's to make input example atleast 10 frames long"""
    batch = np.copy(mini_batch)
    max_len = 10 # max 10 frames
    feat_len = 128 # 128-d embeddings
    seq_len = batch.shape[0]
    if seq_len != max_len:
        batch = np.vstack([batch, np.zeros((max_len - seq_len, feat_len), dtype=batch.dtype)])
    return batch[None,:]


if __name__ == '__main__':
    # Instantiate the VGGish, AudioEventDetector classifier and AudioRecorder objects
    vggish = VGGish()
    aed = AudioEventDetector('models/aed_model_0714_014821.val-acc-0.6995.h5')
    ar = AudioRec()

    while True:
        ar.listen("microphone-results.wav")
        features = vggish.extract_features("microphone-results.wav")

        if features is not None:
            print ("Extracted features of shape: ", features.shape)
            features = pad_sequences(features)
            scores = aed.predict(features)
            pred = np.argsort(scores, axis=1)[0][-3:][::-1]
            pred_classes = [cfg.CLASSES[class_mapping[i]] for i in pred]
            print ("Top 3 predicted classes are: ", pred_classes)
        time.sleep(0.1)

    del vggish

