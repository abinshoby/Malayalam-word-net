# -*- coding: utf-8 -*-
#By abin and hari
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_yaml
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re
import os
from keras.models import model_from_json
import nltk
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


def predict(text):
    yaml_file = open('/home/abin/PycharmProjects/ICFOSS/query_model_test.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    MAX_SEQUENCE_LENGTH = 100
    MAX_NB_WORDS = 20000  # 20000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2
    texts = []
   # text = BeautifulSoup(text)
    texts.append(clean_str(text))#text.get_text()


    tokenizer = Tokenizer(nb_words=20000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #sequences=enc(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    GLOVE_DIR = "/home/abin/PycharmProjects/ICFOSS/glove/"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'wiki.ml.vec'))  # wiki.ml.vec
    for line in f:
        values = line.split()
        word = values[0]
        #print(word)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()  # stores the word and corresponding vector values in a dictionary
    #print(embeddings_index)
    zero_v = []
    actual=[]

    for v in range(300):
        zero_v.append(0)
    for sent in texts:
        #print(sent)
        insent=[]
        to=nltk.word_tokenize(sent)
        for word in to:
            if(word in embeddings_index.keys()):
                #print(word)
                insent.append(embeddings_index[word])
            else:
                insent.append(zero_v)
        actual.append(insent)

    dd=np.asarray(sent2vec(actual))
    print(dd)
    result = model.predict(dd)
    print(result)
    res_list=list(result[0])
    ind=res_list.index(max(res_list))
    print(ind)
    #print("{} positive, {} negeative.".format(result[0,1], result[0,0]))
    data_train = pd.read_csv('/home/abin/PycharmProjects/ICFOSS/data/querydata_to_be_suggested.tsv', sep='\t')
    return (data_train.loc[data_train['sentiment'] == ind]['review']).values.tolist()





def load_map():
    import json
    with open('word_map.json', 'r') as fp:
        word_map = json.load(fp)
    #print(word_map)
    return word_map
def predict2(text):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    t = Tokenizer()
    t.fit_on_texts(text)
    #print(text)
    tok_docs = []





    tok = nltk.word_tokenize(text[0])
    tok_docs.append(tok)
    #print(tok_docs)
    word_map=load_map()
    max1=max(word_map.values())
    text_seq=[]
    for sent in tok_docs:
        s1=[]
        for w in sent:
            if w in word_map.keys():
                s1.append(word_map[w])
            else:
                max1=max1+1
                s1.append(max1)
                word_map[w]=max1
        text_seq.append(s1)
    #print(text_seq)
    #encoded_docs.append(text_seq[0])
    max_length = 10
    padded_docs = pad_sequences(text_seq, maxlen=max_length, padding='post')


    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    # encoded_docs = t.texts_to_sequences(text)
    # print(encoded_docs)
    # pad documents to a max length of 4 words
    # max_length = 4
    # padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    pred = model.predict(padded_docs)
    print(pred)
    print(padded_docs)
    out = []
    for doc in pred:
        out.append(doc.tolist().index(max(doc.tolist())))
    print(out)
    data_train = pd.read_csv('/home/abin/PycharmProjects/ICFOSS/data/querydata_to_be_suggested.tsv', sep='\t')
    return (data_train.loc[data_train['sentiment'] == out[0]]['review']).values.tolist()
# l=predict("ഹോട്ടലുകളുടെ ")
# print(l)
print(predict2(['നിങ്ങൾ ഹോട്ടലുകളുടെ പട്ടിക നൽകാൻ കഴിയുമോ']))