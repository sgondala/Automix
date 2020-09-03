import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import *
import torch.utils.data as Data
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def read_csv_and_return_x_y(data_path, dataset_type='ag_news'): 
    train_df = pd.read_csv(data_path, header=None)
    train_df = train_df.dropna()
    # Here we only use the bodies and removed titles to do the classifications
    if dataset_type=="imdb":
        assert False
    elif dataset_type=="ag_news":
        # Converting 1-4 to 0-3
        y = np.array(train_df[0].values) - 1
        X = np.array(train_df[2].values)
    elif dataset_type=="yahoo_answers":
        y = np.array(train_df[0].values) - 1
        X = np.array(train_df[2].values)
    return X, y

def get_datasets(data_path, 
        max_seq_len=256, 
        model='distilbert-base-uncased',
        train_aug=False, 
        dataset_type='ag_news',
        stratified_split_k=5, 
        percentage_of_val_in_each_split = 0.2):
    """
    Read data, split the dataset, and build dataset for dataloaders.
    """
    # Load the tokenizer for bert
    tokenizer = AutoTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path, header=None)
    train_df = train_df.dropna()
    
    # Here we only use the bodies and removed titles to do the classifications
    n_labels = 0
    if dataset_type=="imdb":
        y = train_df[1].values
        y = np.array([1 if val=='positive' else 0 for val in y])
        X = np.array(train_df[0].values)
    elif dataset_type=="ag_news":
        # Converting 1-4 to 0-3
        y = train_df[0].values
        y = np.array(y) - 1
        X = np.array(train_df[1].values)
        n_labels = 4

    # Split the labeled training set, unlabeled training set, development set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the dataset class for each set
    train_dataset = create_dataset(X_train, y_train, tokenizer, max_seq_len)
    val_dataset = create_dataset(X_val, y_val, tokenizer, max_seq_len)

    return train_dataset, val_dataset, n_labels

class create_dataset(Dataset):
    def __init__(self, dataset_text, dataset_label, tokenizer_type, max_seq_len, aug=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            assert False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text[idx]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        attention_mask = [1] * length + [0] * (self.max_seq_len - length)
        return (torch.tensor(encode_result), 
                torch.tensor(attention_mask),
                self.labels[idx], 
                length)

if __name__ == '__main__':
    pass
    # train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/IMDB_Dataset.csv', imdb=True)
    # train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/ag_news_csv/train.csv')
    # print(len(train_dataset))
    # print(train_dataset[10])
