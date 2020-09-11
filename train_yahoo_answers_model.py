import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.read_data import *
from FastAutoAugment.classification_models.TestClassifier import *
from FastAutoAugment.classification_models.BertBasedClassifier import *

import pickle
import wandb

if __name__ == "__main__":
    seed_everything(42)

    checkpoint_directory = 'yahoo_answers_0.2_base_train'
    wandb_logger = WandbLogger(name=checkpoint_directory, project='autoaugment')

    train = pickle.load(open('data/yahoo_answers_0.2_percent/yahoo_answers_train.pkl', 'rb'))
    val = pickle.load(open('data/yahoo_answers_0.2_percent/yahoo_answers_val.pkl', 'rb'))

    model_name = 'distilbert-base-uncased'

    train_dataset = create_dataset(train['X'], train['y'], model_name, 256)
    val_dataset = create_dataset(val['X'], val['y'], model_name, 256)

    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=3)

    early_stopping = EarlyStopping('val_accuracy', patience=3, mode='max')
    model_checkpoint = ModelCheckpoint( 
        monitor='val_accuracy',
        mode='max',
        save_top_k=1)

    trainer = pl.Trainer(deterministic=True, 
        weights_save_path=f'checkpoints/yahoo_answers_0.2_base_train/',
        logger=wandb_logger, 
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        # distributed_backend='dp',
        gpus=[1],
        gradient_clip_val=0.5)

    model = BertBasedClassifier(model_name=model_name, num_labels=10)
    trainer.fit(model, train_dataloader, val_dataloader)