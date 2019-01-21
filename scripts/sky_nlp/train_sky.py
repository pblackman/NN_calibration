from gensim.models import KeyedVectors
import keras
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.initializers import Constant
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, Embedding
from keras.layers import Dropout, SpatialDropout1D, Bidirectional, GlobalMaxPooling1D,CuDNNLSTM, Lambda, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from load_data_sky import load_data
import numpy as np
from os import path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import sys

from keras import backend as K
import tensorflow as tf

NUM_PARALLEL_EXEC_UNITS = 6

config = tf.ConfigProto(intra_op_parallelism_threads = NUM_PARALLEL_EXEC_UNITS, 
         inter_op_parallelism_threads = 1, 
         allow_soft_placement = True, 
         device_count = {'GPU': NUM_PARALLEL_EXEC_UNITS })

session = tf.Session(config=config)

K.set_session(session)

import os

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"




# Imports to get "utility" package


#sys.path.append( path.abspath("scripts") ) 
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model



print("Evaluate Sky data.")

VECTORS_PATH = "../../data/data_sky/wiki.pt.vec"
EMBEDDING_DIM = 300

(train_posts, train_tags), (test_posts, test_tags) = load_data()
print("Data loaded.")

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 30


tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS, char_level=False)
tokenizer.fit_on_texts(train_posts) # only fit on train
train_sequences = tokenizer.texts_to_sequences(train_posts)
test_sequences = tokenizer.texts_to_sequences(test_posts)
x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Tokenization done.")

word_index = tokenizer.word_index
encoder = LabelEncoder()
encoder.fit(np.concatenate((train_tags, test_tags), axis=0))
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)
y_train = np_utils.to_categorical(np.asarray(y_train))
y_test = np_utils.to_categorical(np.asarray(y_test))

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Creating the model

pt_model = KeyedVectors.load_word2vec_format(VECTORS_PATH)
print("Word vectors loaded.")
vocab = pt_model.vocab
embeddings = np.array([pt_model.word_vec(k) for k in vocab.keys()])

words = []
for word in pt_model.vocab:
    words.append(word)

num_words = len(word_index) + 1

print("Word vocabulary has %s words." % num_words )

not_found = 0;
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word,i in word_index.items():
  try:
    print(word)
    embedding_vector = pt_model[word]
  except:
    not_found+=1
  if embedding_vector is not None:
    # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector

print('%s tokens not found in vocabulary.' % not_found)

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            mask_zero=False,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(LSTM(64, return_sequences=False))(embedded_sequences)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
preds = Dense(len(encoder.classes_), activation='softmax')(x)

model = Model(sequence_input, preds)

print('Model built.')

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.01, clipnorm=.25, beta_1=0.7, beta_2=0.99),
              metrics=['accuracy'])

print('Model compiled.')

hist = model.fit(x_train, y_train,
          batch_size=100,
          epochs=1,
          verbose=2,
          validation_split=0.1,
          shuffle=True)

model.save_weights('model_weight_sky.hdf5')
