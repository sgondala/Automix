import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pandas as pd
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.read_data import *
from FastAutoAugment.classification_models.TestClassifier import *
from FastAutoAugment.classification_models.BertBasedClassifier import *
from FastAutoAugment.classification_models.RandomDistilBertClassifier import *
from FastAutoAugment.classification_models.MixText import *

import pickle
import wandb
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=200,
                    help='number of labeled data')


parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')


parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[7,9,12], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=2, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')

args = parser.parse_args()

# Seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    wandb.init(project="auto_augment")
    
    run_name = 'train_yahoo_on_mixtext_10_per_class_tmix'
    wandb.run.name = run_name
    wandb.run.save()

    train = pickle.load(open('data/paper_yahoo_split/yahoo_train_10_per_class.pkl', 'rb'))
    val = pickle.load(open('data/paper_yahoo_split/yahoo_val_200_per_class.pkl', 'rb'))

    model_name = 'bert-base-uncased'

    train_dataset = create_dataset(train['X'], train['y'], model_name, 256)
    val_dataset = create_dataset(val['X'], val['y'], model_name, 256)

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=3, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=3, shuffle=False)

    model = MixText(10, True).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ]
    )

    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0

    for epoch in range(args.epochs):

        # Train loop
        model.train()
        for batch in train_dataloader:
            encoded_text, _, labels, _ = batch
            logits = model(encoded_text.cuda())
            loss = criterion(logits, labels.cuda())
            wandb.log({'Train loss': loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Val loop
        model.eval()
        with torch.no_grad():
            loss_total = 0
            correct = 0
            total_sample = 0

            for batch in val_dataloader:
                encoded_text, _, target, _ = batch
                
                outputs = model(encoded_text.cuda())
                loss = criterion(outputs, target.cuda())
                # all_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (np.array(predicted.cpu()) == np.array(target.cpu())).sum()
                loss_total += loss.item() * encoded_text.shape[0]
                total_sample += encoded_text.shape[0]
            
            acc_total = correct / total_sample
            loss_total = loss_total / total_sample
            print(f'Epoch number {epoch} Val Loss {loss_total} Val accuracy {acc_total}')
            wandb.log({'Val loss' : loss_total})
            wandb.log({'Val accuracy': acc_total})
        
        if acc_total > best_val_accuracy:
            wandb.run.summary['Val accuracy'] = acc_total
            checkpoint_path = f'checkpoints/{run_name}/'
            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path)
            torch.save(model.state_dict(), f'{checkpoint_path}/model_best.pth')