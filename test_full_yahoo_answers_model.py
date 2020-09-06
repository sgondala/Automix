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
    model = BertBasedClassifier.load_from_checkpoint('checkpoints/full_yahoo_answers_classifier_baseline_without_wandb/lightning_logs/version_3/checkpoints/epoch=0.ckpt')
    # model = torch.load('checkpoints/full_yahoo_answers_classifier_baseline_without_wandb/lightning_logs/version_1/checkpoints/epoch=0.ckpt')
    # trainer.test(test_dataloaders=val_dataloader, ckpt_path='checkpoints/full_yahoo_answers_classifier_baseline_without_wandb/lightning_logs/version_1/checkpoints/epoch=0.ckpt')
