import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import *
import torch.utils.data as Data
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.nlp_augmentations import synonym_replacement_transform, random_insertion_transform, random_swap_transform, random_deletion_transform

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
    def __init__(self, dataset_text, dataset_label, 
            tokenizer_type, max_seq_len=256, mix=None, num_classes=10, alpha=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        self.mix = mix
        self.num_classes = num_classes
        if mix == 'TMix_with_EDA':
            assert alpha != -1, 'Assign alpha with TMix_with_EDA'
            self.augmentations = [synonym_replacement_transform, random_insertion_transform, random_swap_transform, random_deletion_transform]
            self.alpha = alpha

    def __len__(self):
        return len(self.labels)

    def encode_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def prepare_data(self, idx):
        text = self.text[idx]
        # tokens = self.tokenizer.tokenize(text)
        # if len(tokens) > self.max_seq_len:
        #     tokens = tokens[:self.max_seq_len]
        # length = len(tokens)
        # encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        # padding = [0] * (self.max_seq_len - len(encode_result))
        # encode_result += padding
        encode_result, length = self.encode_text(text)
        attention_mask = [1] * length + [0] * (self.max_seq_len - length)
        return (torch.tensor(encode_result), 
                torch.tensor(attention_mask),
                self.labels[idx], 
                length)
    
    def __getitem__(self, idx):
        data_for_idx = self.prepare_data(idx)
        if self.mix is None:
            return data_for_idx
        if self.mix == 'TMix':
            random_index = np.random.randint(0, len(self.labels))
            data_for_random_idx = self.prepare_data(random_index)

            # Combine both
            label_1 = [0]*self.num_classes
            label_1[data_for_idx[2]] = 1
            label_2 = [0]*self.num_classes
            label_2[data_for_random_idx[2]] = 1

            encoded_1 = data_for_idx[0]
            encoded_2 = data_for_random_idx[0]
            # length = max(data_for_idx[3], data_for_random_idx[3])
            # attention_mask_1 = data_for_idx[1]
            # attention_mask_2 = data_for_random_idx[1]
            return (encoded_1, encoded_2, torch.Tensor(label_1), torch.Tensor(label_2))
        
        if self.mix == 'TMix_with_EDA':
            transform_index = np.random.randint(0, len(self.augmentations))
            transform = self.augmentations[transform_index]
            augmented_sentence = transform(self.text[idx], self.alpha, 1)[0]
            encoded_1 = data_for_idx[0]
            encoded_2, _ = self.encode_text(augmented_sentence)
            label_1 = [0]*self.num_classes
            label_1[data_for_idx[2]] = 1
            # label_2 = label_1.copy()
            return (encoded_1, torch.tensor(encoded_2), torch.Tensor(label_1), torch.Tensor(label_1))

        assert False


if __name__ == '__main__':
    pass
    # train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/IMDB_Dataset.csv', imdb=True)
    # train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/ag_news_csv/train.csv')
    # print(len(train_dataset))
    # print(train_dataset[10])
