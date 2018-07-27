# -*- coding: utf-8 -*-
#By abin and hari
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND'] = 'theano'
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def avg(av,sent):
    l=[]
    for i in range(300):
        l.append((av[i]+sent[i])/2)
    return l
def sent2vec(sentgroup):
    last=[]
    for sent in sentgroup:
        av=np.asarray(sent[0])
        for i in range(1,len(sent)):
                av=avg(av,sent[i])
        last.append(av)
    #print("last",last)
    return last







def save(encoded_docs,docs):
    tok_docs=[]
    tokenizer = Tokenizer()
    for d in docs:
        tok=nltk.word_tokenize(d)
        tok_docs.append(tok)

    print(tok_docs)
    word_map={}
    for (doc,seq) in zip(tok_docs,encoded_docs):
        for (w,n) in zip(doc,seq):
            word_map[w]=n
    import json

    with open('word_map.json', 'w') as fp:
        json.dump(word_map, fp)

    #print(word_map)




def train():
    from numpy import array
    from numpy import asarray
    from numpy import zeros
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers import Embedding
    # define documents
    docs=[]
    labels=[]
    data_train = pd.read_csv('/home/abin/PycharmProjects/ICFOSS/data/querydata.tsv', sep='\t')
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        docs.append(clean_str(text.get_text()))  # texts contains list of reviews
        labels.append(data_train.sentiment[idx])  # contains labels corresponding to the reviews [0,1]

    # docs = ['നിങ്ങൾ ഹോട്ടലുകളുടെ പട്ടിക നൽകാൻ കഴിയുമോ',
    #         'ഈ പ്രദേശത്തെ ഹോട്ടലുകളുടെ പട്ടിക ദയവായി എനിക്ക് തരൂ',
    #         'മികച്ച കാഴ്ചകൾ',
    #         'താമസിക്കാൻ പറ്റിയ സ്ഥലം',
    #         'വിനോദസഞ്ചാര കേന്ദ്രങ്ങളുടെ പട്ടിക തരുമോ',
    #         'വിനോദസഞ്ചാര കേന്ദ്രങ്ങളുടെ പട്ടിക',
    #         'താമസിക്കാൻ ഒരു നല്ല സ്ഥലം കണ്ടെത്തുക',
    #         'താമസിക്കാൻ ചില സ്ഥലം നിർദ്ദേശിക്കാനാകുമോ',
    #         'താമസിക്കാൻ ചില നല്ല സ്ഥലങ്ങൾ കണ്ടെത്തുക',
    #         'ഇവിടുത്തെ വിനോദ സഞ്ചാര കേന്ദ്രങ്ങളുടെ പട്ടിക',
    #         'വിനോദസഞ്ചാര കേന്ദ്രങ്ങൾ',
    #         'ബോട്ടിങ്ങിനു പറ്റിയ സ്ഥലങ്ങൾ',
    #         'എവിടെ ബോട്ടിംഗ് ചെയ്യാൻ  പറ്റും',
    #         'എനിക്ക് എവിടെ ബോട്ടിംഗ് ചെയ്യാൻ പറ്റും',
    #         'ഇവിടുത്തെ മികച്ച കാഴ്ചകൾ എന്തൊക്കെ',
    #         'കാഴ്ചകൾ.']
    # define class labels
    # labels = array([4, 4, 3, 2, 0, 0, 2, 2, 2, 0,0,1,1,1,3,3])
    labels = to_categorical(np.asarray(labels))
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)

    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)
    save(encoded_docs,docs)
    # pad documents to a max length of 4 words
    max_length = 10
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('/home/abin/PycharmProjects/ICFOSS/glove/wiki.ml.vec')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # define model
    model = Sequential()
    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=10, trainable=False)
             #no of distinct words,output dim,weight,max no of words in a sentence
    model.add(e)
    #model.add(Flatten())
    model.add(Bidirectional(LSTM(100,return_sequences=True)))
    model.add(Bidirectional(LSTM(100)))
    # model.add(Flatten())
    model.add(Dense(len(labels[0]), activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    print(padded_docs)
    pred=model.predict(padded_docs)
    print(pred)
    out=[]
    for doc in pred:
        out.append(doc.tolist().index(max(doc.tolist())))
    print(out)




train()
