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
from transformers import DistilBertForSequenceClassification, DistilBertConfig, AdamW
from FastAutoAugment.read_data import *

class RandomDistilBertClassifier(LightningModule):
    def __init__(self, num_labels=10, lr=2e-5):
        super().__init__()
        self.lr = lr
        self.config = DistilBertConfig(num_labels=num_labels)
        self.model = DistilBertForSequenceClassification(self.config)
        self.save_hyperparameters()
    
    def forward(self, inputs, labels):
        out = self.model(**inputs, labels=labels)
        return out
    
    def training_step(self, batch, batch_idx):
        inputs, attention_mask, labels, _ = batch
        out = self({'input_ids':inputs, 'attention_mask':attention_mask}, labels)
        loss = out[0] / len(batch)
        logits = out[1]
        labels_predicted = torch.argmax(logits, dim=1)
        assert labels_predicted.shape == labels.shape
        accuracy = (labels_predicted == labels).float().mean()
        logs = {'train_loss': loss.item(), 'train_accuracy': accuracy.item()}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        inputs, attention_mask, labels, _ = batch
        out = self({'input_ids':inputs, 'attention_mask':attention_mask}, labels)
        loss = out[0] / len(batch)
        logits = out[1]
        labels_predicted = torch.argmax(logits, dim=1)
        accuracy = (labels_predicted == labels).float().mean()
        return {'loss': loss, 'accuracy':accuracy}
    
    def validation_end(self, outputs):
        # To deal with last item not being same size as first
        avg_loss = torch.stack([x['loss'] for x in outputs[:-1]]).mean() 
        accuracy = torch.stack([x['accuracy'] for x in outputs[:-1]]).mean()
        logs = {'val_loss': avg_loss, 'val_accuracy': accuracy}
        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_end(self, outputs):
        return self.validation_end(outputs)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    pass
    # train_dataset, val_dataset, n_labels = get_datasets('/home/gondi/Documents/MSCS/Research/fast-autoaugment/data/IMDB_Dataset.csv', imdb=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=3)
    # val_dataloader = DataLoader(val_dataset, batch_size=2, num_workers=3)    
    
    # seed_everything(42)
    # trainer = pl.Trainer(max_epochs=2, deterministic=False, weights_save_path='checkpoints/test_simple_classifier/')
    # model = SimpleClassifier()
    # trainer.fit(model, train_dataloader, val_dataloader)
