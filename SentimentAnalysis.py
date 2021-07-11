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
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)
model = tf.keras.Sequential(
    [tf.keras.layers.Embedding(VOCAB_SIZE, 256), tf.keras.layers.Dense(128), tf.keras.layers.Dense(64),
     tf.keras.layers.LSTM(32), tf.keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=2, validation_split=0.2)
results = model.evaluate(test_data, test_labels)
print(results)
word_index = imdb.get_word_index()


def encode_text(text):
    tokens = kp.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]


def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])


def predictExternal(text, external):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = external.predict(pred)
    print(result[0])


text = "This movie was, quite frankly, one of the worst performances in action cinema that I have ever seen in my life. I can't think of any positive thing to say about it. What a waste of my money!"
pr = "That was one of the greatest and most sublime movies i have ever seen in my life"
nr = "This movie sucked. I hated it and wouldn't watch it again. It was one of the worst and most atrocious things I've ever watched"
r = "Certainly the ex-super-spy is an old cliché, and the movie can't quite get away from other cliches as it goes along, from the overly complicated dialogue-based setup, to the unsurprising double-crosses. Indeed, the movie often feels as if it starts after all the good stuff left off. A general low-energy quality keeps things from ever getting very tense or exciting, and the generic title - just what is a Trigger Point anyway? Is it something to do with a gun? This will probably make this movie fade away quietly and painlessly."
predict(pr)
predict(nr)
predict(text)
predict(r)
model_json = model.to_json()
with open("model.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
json_file = open("model.json", "r")
l_m_json = json_file.read()
json_file.close()
loaded_model = model_from_json(l_m_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
results = loaded_model.evaluate(test_data, test_labels)
print(results)
