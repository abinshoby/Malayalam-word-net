# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000#20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def predict(text, model):
    texts = []
    text = BeautifulSoup(text)
    texts.append(clean_str("it is a stupid movie."))#text.get_text()

    tokenizer = Tokenizer(nb_words=20000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = model.predict(data)
    print(result)
    print("{} positive, {} negeative.".format(result[0,1], result[0,0]))

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('/home/abin/PycharmProjects/ICFOSS/data/labeledTrainData1.tsv', sep='\t')
print
data_train.shape

texts = []
labels = []
#
# print(data_train.review)
for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    texts.append(clean_str(text.get_text()))          #texts contains list of reviews
    labels.append(data_train.sentiment[idx])         #contains labels corresponding to the reviews [0,1]
    #print("text data is"+ str(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# print(tokenizer)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
#print(sequences)                                      # sequences is a list of numbers corresponding to each word in the sentence and for every sentence
#
word_index = tokenizer.word_index
#print(word_index)                                       #word_index contains dictionary of words and their corresponding index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#print(data)                                                    #data contains the sequences arranged so that the list contains the max dimension
labels = to_categorical(np.asarray(labels))
#print(labels)                                                  # arrange labels with dim mx2 and right coloum contains the correct labels and first column contains opposite of right columns
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
#print(indices)                             #indices is just an ordered arrangement of index
np.random.shuffle(indices)
#print(indices)                            #unordered collection of index
data = data[indices]
#print(data)                                 #now data is shuffled indirectly
labels = labels[indices]                    #now label is shuffled in same order as data/reviews
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])  # split the data into training and test data


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
#print(len(x_val[0]))
y_val = labels[-nb_validation_samples:]

print('Training and validation set number of positive and negative reviews')
print
y_train.sum(axis=0)      # no of +ve and -ve reviews

print
y_val.sum(axis=0)

GLOVE_DIR = "/home/abin/PycharmProjects/ICFOSS/glove/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()  # stores the word and corresponding vector values in a dictionary
#print(embeddings_index)

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
#print(embedding_matrix)                                    # stores the vector values of words in the reviews
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  #first layer
embedded_sequences = embedding_layer(sequence_input)             #consider this layer as combination of first layer and embedded layer
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)      #typecast embedded sequence to bi-lstm # was 100
preds = Dense(2, activation='softmax')(l_lstm)   #2 output units
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
#
print("model fitting - Bidirectional LSTM")
model.summary()
print(x_train,y_train)
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=3, batch_size=50)
predict(data_train.review[1],model)
# model_yaml = model.to_yaml()
# with open("eng_text_classified_model1.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("eng_model_we1.h5")
# print("Saved model to disk")
# # Attention GRU network
# class AttLayer(Layer):
#     def __init__(self, **kwargs):
#         self.init = initializers.get('normal')
#         # self.input_spec = [InputSpec(ndim=3)]
#         super(AttLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#         # self.W = self.init((input_shape[-1],1))
#         self.W = self.init((input_shape[-1],))
#         # self.input_spec = [InputSpec(shape=input_shape)]
#         self.trainable_weights = [self.W]
#         super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
#
#     def call(self, x, mask=None):
#         eij = K.tanh(K.dot(x, self.W))
#
#         ai = K.exp(eij)
#         weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
#
#         weighted_input = x * weights.dimshuffle(0, 1, 'x')
#         return weighted_input.sum(axis=1)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[-1])
#
#
# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
#
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
# l_att = AttLayer()(l_gru)
# preds = Dense(2, activation='softmax')(l_att)
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])
#
# print("model fitting - attention GRU network")
# model.summary()
# print(x_train,y_train)
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           nb_epoch=3, batch_size=50)
# #model.predict(x_train[0],y_train[0])
# model_yaml = model.to_yaml()
# with open("eng_text_classified_model2.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("eng_model_we2.h5")
# print("Saved model to disk")
