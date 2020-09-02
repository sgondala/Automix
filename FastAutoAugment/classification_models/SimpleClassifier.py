import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.read_data import *

class SimpleClassifier(LightningModule):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
    
    def forward(self, inputs, labels):
        out = self.model(**inputs, labels=labels)
        return out
    
    def training_step(self, batch, batch_idx):
        inputs, attention_mask, labels, lengths = batch
        loss = self({'input_ids':inputs, 'attention_mask':attention_mask}, labels)[0]
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        inputs, attention_mask, labels, lengths = batch
        loss = self({'input_ids':inputs, 'attention_mask':attention_mask}, labels)[0]
        return {'loss': loss}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-5)

if __name__ == "__main__":
    train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/IMDB_Dataset.csv', imdb=True)
    train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=2, num_workers=3)    
    
    seed_everything(42)
    trainer = pl.Trainer(max_epochs=2, deterministic=False, weights_save_path='checkpoints/test_simple_classifier/')
    model = SimpleClassifier()
    trainer.fit(model, train_dataloader, val_dataloader)