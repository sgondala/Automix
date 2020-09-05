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

    # wandb_logger = WandbLogger(name='test_full_yahoo_answers_model_adam_2e5', project='autoaugment')

    full_val_data = pickle.load(open('data/yahoo_answers_v1/yahoo_answers_full_val.pkl', 'rb'))

    model_name = 'distilbert-base-uncased'

    val_dataset = create_dataset(
        full_val_data['X'], full_val_data['y'], model_name, 256)

    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=3)

    trainer = pl.Trainer(deterministic=True, 
        weights_save_path='checkpoints/full_yahoo_answers_classifier_baseline/', 
        # logger=wandb_logger,
        gpus=1)

    trainer.test(val_dataloader)