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
    seed_everything(42)

    augmentations = [synonym_replacement_transform, random_insertion_transform, random_swap_transform, random_deletion_transform]
    best_probabilities = [0.03527196847551313, 1.1168885841491075e-05, 0.24075230116982638, 0.3247342127244468]
    best_num_of_data = [2, 8, 2, 1]

    wandb_name = 'train_yahoo_answers_model_'
    for i in range(len(augmentations)):
        wandb_name += str(best_probabilities[i]) + '_' + str(best_num_of_data[i]) + '_'
    
    wandb_logger = WandbLogger(name=wandb_name, project='autoaugment')

    full_train_data = pickle.load(open('data/yahoo_answers_v1/yahoo_answers_full_train.pkl', 'rb'))
    full_val_data = pickle.load(open('data/yahoo_answers_v1/yahoo_answers_full_val.pkl', 'rb'))

    actual_X = full_train_data['X'].tolist()
    actual_y = full_train_data['y'].tolist()
    all_aug_X = []
    all_aug_y = []

    # print(type(actual_X))
    # assert False

    for i in range(len(augmentations)):
        print(f'Creating dataset {i}')
        new_X, new_y = create_augmented_dataset(actual_X, actual_y, augmentations[i], best_probabilities[i], best_num_of_data[i])
        all_aug_X += new_X
        all_aug_y += new_y

    print('Length of augmentations ', len(all_aug_X))
    
    all_aug_X += actual_X
    all_aug_y += actual_y

    print('Length of train data ', len(all_aug_X))

    model_name = 'distilbert-base-uncased'

    train_dataset = create_dataset(
        all_aug_X, all_aug_y, model_name, 256)
    val_dataset = create_dataset(
        full_val_data['X'], full_val_data['y'], model_name, 256)

    train_dataloader = DataLoader(train_dataset, batch_size=150, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=150, num_workers=4, shuffle=False)

    early_stopping = EarlyStopping('avg_val_loss', patience=5)

    trainer = pl.Trainer(deterministic=True, 
        weights_save_path=f'checkpoints/{wandb_name}/',
        logger=wandb_logger,
        early_stop_callback=early_stopping,
        distributed_backend='dp',
        gpus=[1,2,3])

    model = BertBasedClassifier(model_name=model_name, num_labels=10)
    trainer.fit(model, train_dataloader, val_dataloader)