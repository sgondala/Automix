import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.read_data import *
from FastAutoAugment.classification_models.TestClassifier import *
from FastAutoAugment.classification_models.BertBasedClassifier import *
from FastAutoAugment.nlp_augmentations import *
from create_augmented_dataset import *
from evaluate_dataset_on_model import *

import pickle
import wandb

if __name__ == "__main__":
    # Constants
    # seed_everything(42)
    print('Starting!!!!')
    model_name = 'distilbert-base-uncased'
    full_val_data = pickle.load(open('data/yahoo_answers_v1/yahoo_answers_full_val.pkl', 'rb'))
    actual_X = full_val_data['X']
    actual_y = full_val_data['y']
    augmentations = [synonym_replacement_transform, random_insertion_transform, random_swap_transform, random_deletion_transform]
    
    print('Before loop')
    # Changes every loop
    for _ in range(100):
        probabilities = 0.2 * np.random.random(size=len(augmentations))
        number_of_data = np.random.choice([1,2,4,8],size=len(augmentations), replace=True)
        print('Probabilities ', probabilities)
        print('Number of data ', number_of_data)

        all_aug_X = []
        all_aug_y = []

        for i in range(len(augmentations)):
            print(f'Creating dataset {i}')
            new_X, new_y = create_augmented_dataset(actual_X, actual_y, augmentations[i], probabilities[i], number_of_data[i])
            # print("New x ", new_X)
            # print("New y ", new_y)
            all_aug_X += new_X
            all_aug_y += new_y
        
        print('Length of dataset ', len(all_aug_X))

        # val_dataset = create_dataset(actual_X, actual_y, model_name, 256)
        val_dataset = create_dataset(all_aug_X, all_aug_y, model_name, 256)
        wandb_name = 'random_experiments_'
        for i in range(len(augmentations)):
            wandb_name += str(probabilities[i]) + '_' + str(number_of_data[i]) + '_'

        print('Before evaluate ')
        val_accuracy = evaluate_dataset_on_model(
            wandb_name = wandb_name, 
            checkpoint = 'checkpoints/full_yahoo_answers_classifier_baseline/lightning_logs/version_7/checkpoints/epoch=6.ckpt',
            dataset = val_dataset,
        )[0]['val_accuracy']

        # print('Probabilities ', probabilities)
        # print('Number of data ', number_of_data)
        print('Val accuracy ', val_accuracy)
        print('--------------------')