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
from FastAutoAugment.nlp_augmentations import *
from create_augmented_dataset import *
from evaluate_dataset_on_model import *

import pickle
import wandb
import random

from hyperopt import fmin, tpe, hp, Trials

seed_everything(42)

def optimization_function(args):
    p1, p2, p3, p4, n1, n2, n3, n4 = args
    probabilities = [p1,p2,p3,p4]
    number_of_data = [n1,n2,n3,n4]

    model_name = 'distilbert-base-uncased'
    val = pickle.load(open('data/yahoo_answers_0.2_percent/yahoo_answers_val.pkl', 'rb'))
    
    # length_of_full_data = len(val['X'])
    # indices = list(range(length_of_full_data))
    # random.Random(42).shuffle(indices) # Making sure we get same indices everytime; why not just save?
    # indices = indices[:int(length_of_full_data/2)]

    # actual_X = np.array(val['X'])[indices]
    # actual_y = np.array(val['y'])[indices]
    actual_x = val['X']
    actual_y = val['y']
    
    augmentations = [synonym_replacement_transform, random_insertion_transform, random_swap_transform, random_deletion_transform]

    print('Probabilities ', probabilities)
    print('Number of data ', number_of_data)

    all_aug_x = []
    all_aug_y = []

    for i in range(len(augmentations)):
        # print(f'Creating dataset {i}')
        new_X, new_y = create_augmented_dataset(actual_x, actual_y, augmentations[i], probabilities[i], number_of_data[i])
        all_aug_x += new_X
        all_aug_y += new_y

    all_aug_x += actual_x.tolist()
    all_aug_y += actual_y.tolist()

    # print('Length of dataset ', len(actual_X))

    # val_dataset = create_dataset(actual_X, actual_y, model_name, 256)
    val_dataset = create_dataset(all_aug_x, all_aug_y, model_name, 256)

    wandb_name = 'hyperopt_experiments_yahoo_0.2_percent_with_original_random_distilbert_'
    for i in range(len(augmentations)):
        wandb_name += str(probabilities[i]) + '_' + str(number_of_data[i]) + '_'

    # print('Before evaluate ')
    val_accuracy = evaluate_dataset_on_model(
        wandb_name = wandb_name, 
        checkpoint = 'checkpoints/yahoo_answers_0.2_base_train_random_distilbert/autoaugment/1qu8eci3/checkpoints/epoch=60.ckpt',
        dataset = val_dataset,
        model_type = 'RandomDistilBertClassifier'
    )[0]['val_accuracy']

    return -val_accuracy # - because we're minimizing
    

if __name__ == "__main__":
    trials = Trials()
    
    space = [
        hp.uniform('p1', 0, 0.4),
        hp.uniform('p2', 0, 0.4),
        hp.uniform('p3', 0, 0.4),
        hp.uniform('p4', 0, 0.4),
        hp.choice('n1',[1,2,4,8]),
        hp.choice('n2',[1,2,4,8]),
        hp.choice('n3',[1,2,4,8]),
        hp.choice('n4',[1,2,4,8]),
    ]

    best = fmin(fn=optimization_function,
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials)
    
    pickle.dump(trials, open('trials_hyperopt_yahoo_0.2_percent.pkl', 'wb'))    