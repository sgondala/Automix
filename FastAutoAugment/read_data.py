import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import *
import torch.utils.data as Data
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_datasets(data_path, 
        max_seq_len=256, 
        model='distilbert-base-uncased',
        train_aug=False, 
        imdb=True):
    """
    Read data, split the dataset, and build dataset for dataloaders.
    """
    # Load the tokenizer for bert
    tokenizer = AutoTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path, header=None)
    
    # Here we only use the bodies and removed titles to do the classifications
    y = train_df[1].values
    if imdb:
        y = np.array([1 if val=='positive' else 0 for val in y])
    X = np.array(train_df[0].values)

    n_labels = 2

    # Split the labeled training set, unlabeled training set, development set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the dataset class for each set
    train_dataset = create_dataset(X_train[:10], y_train[:10], tokenizer, max_seq_len)
    val_dataset = create_dataset(X_val[:10], y_val[:10], tokenizer, max_seq_len)

    return train_dataset, val_dataset, n_labels

class create_dataset(Dataset):
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
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
    train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/IMDB_Dataset.csv', imdb=True)
    print(len(train_dataset))
    print(train_dataset[10])