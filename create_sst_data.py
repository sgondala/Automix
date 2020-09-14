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

pos = 1
neg = 0
seed = 42

with open('data/sst2/stsa.binary.train.txt', 'r') as file:
    train_df = pd.read_csv(StringIO(''.join(line[0] + '\t' + line[2:] for line in file.readlines())), sep='\t')
train_df.columns = ['sentiment', 'sentence']
with open('data/sst2/stsa.binary.dev.txt', 'r') as file:
    val_df = pd.read_csv(StringIO(''.join(line[0] + '\t' + line[2:] for line in file.readlines())), sep='\t')
val_df.columns = ['sentiment', 'sentence']
with open('data/sst2/stsa.binary.test.txt', 'r') as file:
    test_df = pd.read_csv(StringIO(''.join(line[0] + '\t' + line[2:] for line in file.readlines())), sep='\t')
test_df.columns = ['sentiment', 'sentence']

subsampled_train, throwaway = train_test_split(train_df, train_size=0.01, random_state=seed)
subsampled_val, throwaway = train_test_split(val_df, train_size=(10/len(val_df)), random_state=seed, stratify=val_df['sentiment'])

pickle.dump(subsampled_train, open('data/sst2/sst2_1_percent_train.pkl', 'wb'))
pickle.dump(subsampled_val, open('data/sst2/sst2_10_samples_val.pkl', 'wb'))
pickle.dump(test_df, open('data/sst2/sst2_all_test.pkl', 'wb'))
