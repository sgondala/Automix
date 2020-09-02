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
import pickle
import wandb

seed_everything(42)

wandb_logger = WandbLogger(name='train_full_ag_model_adam_2e5', project='autoaugment')

full_train_data = pickle.load(open('data/ag_news_v1/ag_news_full_train.pkl', 'rb'))
full_val_data = pickle.load(open('data/ag_news_v1/ag_news_full_val.pkl', 'rb'))

model_name = 'distilbert-base-uncased'

train_dataset = create_dataset(
    full_train_data['X'], full_train_data['y'], model_name, 256)
val_dataset = create_dataset(
    full_val_data['X'], full_val_data['y'], model_name, 256)

train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=3)
val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=3)

early_stopping = EarlyStopping('avg_val_loss')

trainer = pl.Trainer(deterministic=True, weights_save_path='checkpoints/full_ag_classifier_baseline/', early_stop_callback=early_stopping)

model = SimpleClassifier(model_name=tokenizer_type, num_labels=4)
trainer.fit(model, train_dataloader, val_dataloader)