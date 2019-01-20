import keras
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from load_data_sky import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.abspath("scripts") ) 
from utility.evaluation import evaluate_model

from gensim.models import KeyedVectors

VECTORS_PATH = r"C:\Users\Patrick\Documents\mestrado\machine-learning\sky-data\wiki.pt.vec"


(train_posts, train_tags), (test_posts, test_tags) = load_data()

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 30


tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS, char_level=False)
tokenizer.fit_on_texts(train_posts) # only fit on train
train_sequences = tokenizer.texts_to_sequences(train_posts)
test_sequences = tokenizer.texts_to_sequences(test_posts)
x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
encoder = LabelEncoder()
encoder.fit(np.concatenate((train_tags, test_tags), axis=0))
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)
y_train = np_utils.to_categorical(np.asarray(y_train))
y_test = np_utils.to_categorical(np.asarray(y_test))

# Creating the model

pt_model = KeyedVectors.load_word2vec_format(VECTORS_PATH)


print('teste')