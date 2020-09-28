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
from io import StringIO

seed = 42

with open('../data/sst2/stsa.binary.train.txt', 'r') as file:
    train_df = pd.read_csv(StringIO(''.join(line[0] + '\t' + line[2:] for line in file.readlines())), sep='\t')
train_df.columns = ['sentiment', 'sentence']
train_df.dropna()
with open('../data/sst2/stsa.binary.dev.txt', 'r') as file:
    val_df = pd.read_csv(StringIO(''.join(line[0] + '\t' + line[2:] for line in file.readlines())), sep='\t')
val_df.columns = ['sentiment', 'sentence']
val_df.dropna()
with open('../data/sst2/stsa.binary.test.txt', 'r') as file:
    test_df = pd.read_csv(StringIO(''.join(line[0] + '\t' + line[2:] for line in file.readlines())), sep='\t')
test_df.columns = ['sentiment', 'sentence']
test_df.dropna()

subsampled_train, throwaway = train_test_split(train_df, train_size=0.01, random_state=seed)
subsampled_val, throwaway = train_test_split(val_df, train_size=(10/len(val_df)), random_state=seed, stratify=val_df['sentiment'])

X_train = np.array(subsampled_train['sentence'])
y_train = np.array(subsampled_train['sentiment'])
X_val = np.array(subsampled_val['sentence'])
y_val = np.array(subsampled_val['sentiment'])
X_test = np.array(test_df['sentence'])
y_test = np.array(test_df['sentiment'])

train_data = {'X': X_train, 'y': y_train}
val_data = {'X': X_val, 'y': y_val}
test_data = {'X': X_test, 'y': y_test}

pickle.dump(train_data, open('../data/sst2/sst2_1_percent_train.pkl', 'wb'))
pickle.dump(val_data, open('../data/sst2/sst2_10_samples_val.pkl', 'wb'))
pickle.dump(test_data, open('../data/sst2/sst2_all_test.pkl', 'wb'))
