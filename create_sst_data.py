import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import *
import torch.utils.data as Data
import pickle
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.read_data import *
import pickle

train_val_test_df = pd.read_csv('data/stanfordSentimentTreebank/datasetSplit.csv')
full_train_indexes = []
full_val_indexes = []
full_test_indexes = []
[(full_train_indexes.append(row[0])) for row in train_val_test_df.to_numpy() if row[1] == 1]
[(full_val_indexes.append(row[0])) for row in train_val_test_df.to_numpy() if row[1] == 3]
[(full_test_indexes.append(row[0])) for row in train_val_test_df.to_numpy() if row[1] == 2]
all_sentences = pd.read_csv('data/stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
all_sentences = all_sentences["sentence"]
all_train = []
all_val = []
all_test = []
for index in range(0, len(all_sentences)):
    if (index + 1) in full_train_indexes:
        all_train.append(all_sentences[index])
    elif (index + 1) in full_val_indexes:
        all_val.append(all_sentences[index])
    else:
        all_test.append(all_sentences[index])

subsampled_train = pd.DataFrame(random.sample(all_train, round(0.01 * len(all_train))))
subsampled_val = pd.DataFrame(random.sample(all_val, 10))
all_test = pd.DataFrame(all_test)