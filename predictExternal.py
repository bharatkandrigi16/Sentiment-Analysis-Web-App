# coding=utf-8
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import keras_preprocessing as kp
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import model_from_json
import numpy as np

BATCH_SIZE = 64
MAXLEN = 250
VOCAB_SIZE = 88584
word_index = imdb.get_word_index()


def encode_text(text):
    tokens = kp.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]


def predictExternal(text, external):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = external.predict(pred)
    return result[0]



