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

from FastAutoAugment.classification_models.RandomDistilBertClassifier import RandomDistilBertClassifier
from FastAutoAugment.read_data import *
from FastAutoAugment.classification_models.TestClassifier import *
from FastAutoAugment.classification_models.BertBasedClassifier import *

import pickle
import wandb

if __name__ == "__main__":
    seed_everything(42)

    checkpoint_directory = 'sst2_base_train_random_distilbert'
    wandb_logger = WandbLogger(name=checkpoint_directory, project='autoaugment')

    train = pickle.load(open('../data/sst2/sst2_1_percent_train.pkl', 'rb'))
    val = pickle.load(open('../data/sst2/sst2_10_samples_val.pkl', 'rb'))
    print(train)

    model_name = 'bert-base-uncased'

    train_dataset = create_dataset(train['X'], train['y'], model_name, 256, num_classes=2)
    val_dataset = create_dataset(val['X'], val['y'], model_name, 256, num_classes=2)

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=3, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=3, shuffle=False)

    early_stopping = EarlyStopping('val_accuracy', patience=6, mode='max')
    model_checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=1)

    trainer = pl.Trainer(deterministic=True,
                         weights_save_path=f'checkpoints/{checkpoint_directory}/',
                         logger=wandb_logger,
                         early_stop_callback=early_stopping,
                         checkpoint_callback=model_checkpoint,
                         distributed_backend='dp',
                         gpus=None,
                         # gradient_clip_val=0.5,
                         num_sanity_val_steps=-1,
                         min_epochs=100)

    # model = BertBasedClassifier(model_name=model_name, num_labels=10)
    model = RandomDistilBertClassifier(num_labels=2, lr=2e-5)
    trainer.fit(model, train_dataloader, val_dataloader)
