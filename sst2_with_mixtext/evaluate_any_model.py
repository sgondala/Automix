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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--checkpoint-path', type=str, default='',
                    help='Saved model checkpoint')

args = parser.parse_args()

# Seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    wandb.init(project="auto_augment")
    wandb.config.update(args)
    
    run_name = f'test_{args.checkpoint_path}'
    wandb.run.name = run_name
    wandb.run.save()

    test = pickle.load(open('../data/sst2/sst2_all_test.pkl', 'rb'))

    model_name = 'bert-base-uncased'

    test_dataset = create_dataset(test['X'], test['y'], model_name, 256, mix=None)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=3, shuffle=False)

    # model = MixText(10, True).cuda()
    # saved_checkpoint = torch.load(args.checkpoint_path)
    # model.load_state_dict(saved_checkpoint)
    model = torch.load(args.checkpoint_path).cuda()
    # device = torch.device('cuda:0')
    # model.to(device)
    # model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    val_accuracy = 0

    model.eval()
    with torch.no_grad():
        loss_total = 0
        correct = 0
        total_sample = 0

        for batch in tqdm(test_dataloader):
            encoded_text, _, target, _ = batch
            
            outputs = model(encoded_text.cuda())
            loss = criterion(outputs, target.cuda())
            _, predicted = torch.max(outputs.data, 1)
            correct += (np.array(predicted.cpu()) == np.array(target.cpu())).sum()
            loss_total += loss.item() * encoded_text.shape[0]
            total_sample += encoded_text.shape[0]
        
        acc_total = correct / total_sample
        loss_total = loss_total / total_sample
        wandb.log({'Test loss' : loss_total})
        wandb.log({'Test accuracy': acc_total})
        print('Test accuracy ', acc_total)