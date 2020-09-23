import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import *
from pytorch_lightning import *
from pytorch_lightning.loggers import *
from pytorch_lightning.callbacks import *

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
    
    # Best values for full yahoo
    # best_probabilities = [0.03527196847551313, 1.1168885841491075e-05, 0.24075230116982638, 0.3247342127244468]
    # best_num_of_data = [2, 8, 2, 1]

    # Random values for yahoo 0.5%
    # best_probabilities = [0.014, 0.0006, 0.059, 0.006]
    # best_num_of_data = [1, 8, 2, 4]

    # Random values for yahoo 0.2%
    # best_probabilities = [0.10, 0.06, 0.20, 0.05]
    # best_num_of_data = [1, 2, 1, 2]

    # EDA paper values for yahoo 0.2%
    # best_probabilities = [0.05]*4
    # best_num_of_data = [8]*4

    # Values for yahoo 0.2% on untrained distilbert
    best_probabilities = [0.31, 0.04, 0.14, 0.01]
    best_num_of_data = [1, 8, 1, 4]

    wandb_name = 'train_augmented_yahoo_answers_0.2_random_distilbert_'
    for i in range(len(augmentations)):
        wandb_name += str(best_probabilities[i]) + '_' + str(best_num_of_data[i]) + '_'
    
    wandb_logger = WandbLogger(name=wandb_name, project='autoaugment')

    train = pickle.load(open('data/yahoo_answers_0.2_percent/yahoo_answers_train.pkl', 'rb'))
    val = pickle.load(open('data/yahoo_answers_0.2_percent/yahoo_answers_val.pkl', 'rb'))

    actual_X = train['X'].tolist()
    actual_y = train['y'].tolist()

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

    train_dataset = create_dataset(all_aug_X, all_aug_y, model_name, 256)
    val_dataset = create_dataset(val['X'], val['y'], model_name, 256)

    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, num_workers=4, shuffle=False)

    early_stopping = EarlyStopping('val_accuracy', patience=5, mode='max')
    model_checkpoint = ModelCheckpoint( 
        monitor='val_accuracy',
        mode='max',
        save_top_k=1)

    trainer = pl.Trainer(deterministic=True, 
        weights_save_path=f'checkpoints/{wandb_name}/',
        logger=wandb_logger,
        early_stop_callback=early_stopping,
        distributed_backend='dp',
        gpus=[4,5,6,7],
        gradient_clip_val=0.5,
        num_sanity_val_steps=-1,
        # min_epochs=100
    )

    # model = BertBasedClassifier(model_name=model_name, num_labels=10)
    model = RandomDistilBertClassifier(num_labels=10, lr=1e-6)
    trainer.fit(model, train_dataloader, val_dataloader)