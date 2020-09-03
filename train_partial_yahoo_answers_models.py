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

import pickle
import wandb

if __name__ == "__main__":
    seed_everything(42)
    model_name = 'distilbert-base-uncased'

    for i in range(5):
        wandb_logger = WandbLogger(name=f'train_partial_yahoo_answers_model_{i}_adam_2e5', project='autoaugment')

        data = pickle.load(open(f'data/yahoo_answers_v1/train_val_split_part_{i}.pkl', 'rb'))

        train_dataset = create_dataset(
            data['X_train'], data['y_train'], model_name)
        val_dataset = create_dataset(
            data['X_test'], data['y_test'], model_name)

        train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=3)
        val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=3)

        early_stopping = EarlyStopping('avg_val_loss', patience=7)

        trainer = pl.Trainer(deterministic=True, 
            weights_save_path=f'checkpoints/partial_yahoo_answers_classifier_{i}/', early_stop_callback=early_stopping, 
            logger=wandb_logger,
            distributed_backend='ddp',
            gpus=2)

        model = BertBasedClassifier(model_name=model_name, num_labels=10)
        trainer.fit(model, train_dataloader, val_dataloader)
