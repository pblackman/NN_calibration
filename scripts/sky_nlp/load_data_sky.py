import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#DATA_PATH = r"C:\Users\Patrick\Documents\mestrado\machine-learning\sky-data\primeiro-prompt-dataset-hermes-2.43.txt"
DATA_PATH = "../../data/data_sky/primeiro-prompt-dataset-hermes-2.43.txt"

def load_data():
   
    data = pd.read_csv(DATA_PATH, sep="|",engine='python')

    ## remove classes with few instances (less than 2)
    bytag = data.groupby('class').aggregate(np.count_nonzero)
    tags = bytag[bytag.utterance >= 2].index
    data = data[data['class'].isin(tags)]
    labels = data['class']

    train_posts, train_tags, test_posts, test_tags = train_test_split(data['utterance'], data['class'],
                                                        stratify=data['class'], shuffle=True,
                                                        test_size=0.20)
    
    return ((train_posts, train_tags), (test_posts, test_tags))