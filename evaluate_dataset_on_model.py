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

def evaluate_dataset_on_model(wandb_name, checkpoint, dataset):    
    # seed_everything(42)
    # print('Inside evaluate ')
    wandb_logger = WandbLogger(name=wandb_name, project='autoaugment')

    model = BertBasedClassifier.load_from_checkpoint(checkpoint)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=3)

    trainer = pl.Trainer(deterministic=True, 
        logger=wandb_logger, 
        distributed_backend='dp',
        gpus=[2,3])
    result_dict = trainer.test(model = model, test_dataloaders = dataloader)
    return result_dict

if __name__ == "__main__":
    model_name = 'distilbert-base-uncased'
    full_val_data = pickle.load(open('data/yahoo_answers_v1/yahoo_answers_full_val.pkl', 'rb'))
    val_dataset = create_dataset(
        full_val_data['X'], full_val_data['y'], model_name, 256)
    
    print('Final accuracy ', evaluate_dataset_on_model(
        wandb_name = 'test_full_yahoo_answers_model_no_aug', 
        checkpoint = 'checkpoints/full_yahoo_answers_classifier_baseline/lightning_logs/version_7/checkpoints/epoch=6.ckpt',
        dataset = val_dataset
        )['val_accuracy']
    )

def evaluate_sst2_dataset_on_model():
    model_name = 'distilbert-base-uncased'
    full_val_data = pickle.load(open('data/sst2/sst2_10_samples_val.pkl', 'rb'))
    val_dataset = create_dataset(full_val_data['sentiment'], full_val_data['sentence'], model_name, 256)

    print('Final accuracy ', evaluate_dataset_on_model(
        wandb_name='test_full_sst2_model_no_aug',
        checkpoint='checkpoints/sst2_classifier_baseline/lightning_logs/version_7/checkpoints/epoch=6.ckpt',
        dataset=val_dataset)['val_accuracy'])