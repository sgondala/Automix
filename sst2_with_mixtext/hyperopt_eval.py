import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, ConcatDataset

import pandas as pd
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FastAutoAugment.read_data import *
from FastAutoAugment.classification_models.MixText import *

import pickle
import wandb
import argparse
import os
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, Trials

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--checkpoint-path', type=str, default='checkpoints/train_yahoo_on_mixtext_10_per_class_no_augmentations/model_best.pth', help='Saved model checkpoint')

parser.add_argument('--sub-policies-per-policy', type=int, default=3)

parser.add_argument('--number-of-policies-to-evaluate', type=int, default=200)

parser.add_argument('--alpha', type=float, default=2)

parser.add_argument('--mix-layers', nargs='+',
                    default=[7,9,12], type=int, help='define mix layer set')

args = parser.parse_args()

# Seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

functions = ['synonym_replacement', 'random_insert', 'random_swap', 'random_delete', 'inter_lada', 'intra_lada', 'random']
function_name_to_tmix_functions_map = {
    'synonym_replacement': 'TMix_with_EDA_synonym_replacement',
    'random_insert': 'TMix_with_EDA_random_insert',
    'random_swap': 'TMix_with_EDA_random_swap',
    'random_delete': 'TMix_with_EDA_random_delete',
    'random': 'TMix',
    'intra_lada': 'Intra_LADA'
}

def own_loss(logits, target, num_labels):
    assert logits.shape == target.shape
    loss = -torch.sum(F.log_softmax(logits, dim=1)*target, axis=1)
    assert loss.shape[0] == target.shape[0]
    return loss.mean()

def optimization_function(input_arguments):
    next_input_arguments_index = 0

    all_datasets = []
    
    wandb.init(project="auto_augment")
    wandb_name = 'test_sst2_val_5_hyperopt_'

    model_name = 'bert-base-uncased'
    dataset_identifier = 'val_5'

    for i in range(args.sub_policies_per_policy):
        function_name = functions[int(input_arguments[next_input_arguments_index])]
        probability_of_application = input_arguments[next_input_arguments_index + 1]
        argument_1 = input_arguments[next_input_arguments_index + 2]
        argument_2 = input_arguments[next_input_arguments_index + 3]
        wandb_name += function_name + "_" + str(probability_of_application) + "_" + str(argument_1) + "_" + str(argument_2) + "_"
        next_input_arguments_index += 4

        val = pickle.load(open('../data/sst2/sst2_10_samples_val.pkl', 'rb'))

        if function_name in ['random', 'intra_lada']:
            # 0 arguments
            mix = function_name_to_tmix_functions_map[function_name]
            val_dataset = create_dataset(
                val['X'], val['y'], model_name, 256, mix=mix, num_classes=2,
                probability_of_application = probability_of_application, 
                dataset_identifier = dataset_identifier)
        elif function_name in ['synonym_replacement', 'random_insert', 'random_swap', 'random_delete']:
            # one argument
            mix = function_name_to_tmix_functions_map[function_name]
            val_dataset = create_dataset(
                val['X'], val['y'], model_name, 256, mix=mix, num_classes=2,
                probability_of_application = probability_of_application, alpha=argument_1, 
                dataset_identifier = dataset_identifier)
        elif function_name == 'inter_lada':
            # 2 arguments
            knn = int(argument_1 * 10)
            mu = argument_2
            val_dataset = create_dataset(
                val['X'], val['y'], model_name, 256, mix='Inter_LADA', num_classes=2,
                probability_of_application = probability_of_application, 
                knn_lada=knn, mu_lada=mu, dataset_identifier = dataset_identifier)
        else:
            assert False

        all_datasets.append(val_dataset)
    
    wandb.run.name = wandb_name
    wandb.run.save()

    val_dataset_combined = ConcatDataset(all_datasets)
    val_dataloader = DataLoader(val_dataset_combined, batch_size=args.batch_size, num_workers=4)

    base_model = torch.load(args.checkpoint_path).cuda()
    base_model.eval()

    with torch.no_grad():
        loss_total = 0
        total_sample = 0

        for batch in tqdm(val_dataloader):
            encoded_1, encoded_2, label_1, label_2 = batch
            assert encoded_1.shape == encoded_2.shape
            
            mix_layer = np.random.choice(args.mix_layers)
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)
            
            logits = base_model(encoded_1.cuda(), encoded_2.cuda(), l, mix_layer)
            combined_labels = label_1 * l + label_2 * (1-l)
            loss = own_loss(logits, combined_labels.cuda(), num_labels=10)
            loss_total += loss.item() * encoded_1.shape[0]
            total_sample += encoded_1.shape[0]
        
        wandb.log({'Test loss' : loss_total})
        return loss_total

if __name__ == "__main__":
    trials = Trials()
    space = []

    number_of_functions = len(functions)

    for i in range(args.sub_policies_per_policy):
        space.append(hp.choice(f'n_{i}', list(range(number_of_functions))))
        space.append(hp.uniform(f'p_application_{i}', 0, 1))
        space.append(hp.uniform(f'argument_{i}_1', 0, 1))
        space.append(hp.uniform(f'argument_{i}_2', 0, 1))

    best = fmin(fn=optimization_function,
        space=space,
        algo=tpe.suggest,
        max_evals=args.number_of_policies_to_evaluate,
        trials=trials)