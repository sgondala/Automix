import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from FastAutoAugment.read_data import *
from FastAutoAugment.classification_models.MixText import *

import pickle
import wandb
import argparse
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, Trials

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--checkpoint-path', type=str, default='checkpoints/train_yahoo_on_mixtext_10_per_class_no_augmentations/model_best.pth', help='Saved model checkpoint')

parser.add_argument('--sub-policies-per-policy', type=int, default=3)

parser.add_argument('--number-of-policies-to-evaluate', type=int, default=50)

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

def own_loss(logits, target, num_labels):
    assert logits.shape == target.shape
    loss = -torch.sum(F.log_softmax(logits, dim=1)*target, axis=1)
    assert loss.shape[0] == target.shape[0]
    return loss.mean()

def optimization_function(input_arguments):
    arg1, arg2, arg3 = input_arguments

    wandb.init(project="auto_augment", reinit=True)
    wandb_name = f'hyperopt_single_inter_lada_layers_{arg1}_{arg2}_{arg3}'

    model_name = 'bert-base-uncased'
    dataset_identifier = 'val_200'
    
    val = pickle.load(open('data/paper_yahoo_split/yahoo_val_200_per_class.pkl', 'rb'))

    # knn = arg1
    # mu = arg2
    knn = 7
    mu = 0.23

    val_dataset = create_dataset(val['X'], val['y'], model_name, 256, mix='Inter_LADA', num_classes=10,knn_lada=knn, mu_lada=mu, dataset_identifier = dataset_identifier)
    
    wandb.run.name = wandb_name
    wandb.run.save()

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    base_model = torch.load(args.checkpoint_path).cuda()
    base_model.eval()

    with torch.no_grad():
        loss_total = 0
        total_sample = 0

        for batch in tqdm(val_dataloader, desc='Validation loop'):
            encoded_1, encoded_2, label_1, label_2 = batch
            assert encoded_1.shape == encoded_2.shape
            
            # mix_layer = np.random.choice(args.mix_layers)
            mix_layer = np.random.choice([arg1, arg2, arg3])
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)
            
            logits = base_model(encoded_1.cuda(), encoded_2.cuda(), l, mix_layer)
            combined_labels = label_1 * l + label_2 * (1-l)
            loss = own_loss(logits, combined_labels.cuda(), num_labels=10)
            loss_total += loss.item() * encoded_1.shape[0]
            total_sample += encoded_1.shape[0]
        
        loss_total = loss_total/total_sample
        wandb.log({'Test loss' : loss_total})
        print('Test loss ', loss_total)
        return loss_total

if __name__ == "__main__":
    trials = Trials()
    space = []

    # space.append(hp.choice(f'arg1', list(range(1, 10))))
    # space.append(hp.uniform(f'arg2', 0, 1))
    space.append(hp.choice(f'arg1', list(range(1,12))))
    space.append(hp.choice(f'arg2', list(range(1,12))))
    space.append(hp.choice(f'arg3', list(range(1,12))))

    best = fmin(fn=optimization_function,
        space=space,
        algo=tpe.suggest,
        max_evals=args.number_of_policies_to_evaluate,
        trials=trials)
    
    pickle.dump(
        trials, 
        open(f'data/saved_logs/hyperopt_single_inter_lada_layers_changes_{args.number_of_policies_to_evaluate}.pkl', 'wb'))