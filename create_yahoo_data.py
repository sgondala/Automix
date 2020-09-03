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

seed=42

X, y = read_csv_and_return_x_y('data/yahoo_answers_csv/train.csv', 'yahoo_answers')
X_use, _, y_use,_ = train_test_split(X, y, test_size=0.8, random_state=seed, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_use, y_use, test_size=0.25, random_state=seed, stratify=y_use)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)

val_data = {}
val_data['X'] = X_val
val_data['y'] = y_val

train_data = {}
train_data['X'] = X_train
train_data['y'] = y_train

print(len(X_train))
print(len(X_val))
pickle.dump(train_data, open('data/yahoo_answers_v1/yahoo_answers_full_train.pkl', 'wb'))
pickle.dump(val_data, open('data/yahoo_answers_v1/yahoo_answers_full_val.pkl', 'wb'))

datasets = []
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
splits = sss.get_n_splits(X_train, y_train)

i = 0
for train_index, test_index in sss.split(X_train, y_train):
    dataset_here = {}
    dataset_here['X_train'] = X_train[train_index]
    dataset_here['y_train'] = y_train[train_index]
    dataset_here['X_test'] = X_train[test_index]
    dataset_here['y_test'] = y_train[test_index]
    assert len(dataset_here['X_train']) == len(dataset_here['y_train'])
    assert len(dataset_here['X_test']) == len(dataset_here['y_test'])
    print(len(train_index), len(test_index))
    pickle.dump(dataset_here, open(f'data/yahoo_answers_v1/train_val_split_part_{i}.pkl', 'wb'))
    i += 1