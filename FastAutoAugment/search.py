import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch

import numpy as np
from hyperopt import hp
import ray
import gorilla
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments
from tqdm import tqdm

from FastAutoAugment.archive import remove_deplicates, policy_decoder
from FastAutoAugment.augmentations import augment_list
from FastAutoAugment.common import get_logger, add_filehandler
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.metrics import Accumulator
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.train import train_and_eval
from theconf import Config as C, ConfigArgumentParser
from FastAutoAugment.opts import parse_search_opts

import json
from pystopwatch2 import PyStopwatch

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

top1_valid_by_cv = defaultdict(lambda: list)

def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
    print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)

patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)

logger = get_logger('Fast AutoAugment')

def _get_path(dataset, model, tag):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/%s_%s_%s.model' % (dataset, model, tag))

# @ray.remote(num_gpus=4, max_calls=1)
@ray.remote(num_cpus=4, max_calls=1)
def train_model(config, dataroot, augment, cv_ratio_test, cv_fold, save_path=None, skip_exist=False):
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    result = train_and_eval(None, dataroot, cv_ratio_test, cv_fold, save_path=save_path, only_eval=skip_exist)
    return C.get()['model']['type'], cv_fold, result


def eval_tta(config, augment, reporter):
    C.get()
    C.get().conf = config
    cv_ratio_test, cv_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']

    # setup - provided augmentation rules
    C.get()['aug'] = policy_decoder(augment, augment['num_policy'], augment['num_op'])

    # eval
    model = get_model(C.get()['model'], num_class(C.get()['dataset']))
    ckpt = torch.load(save_path)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    loaders = []
    for _ in range(augment['num_policy']):  # TODO
        _, tl, validloader, tl2 = get_dataloaders(C.get()['dataset'], C.get()['batch'], augment['dataroot'], cv_ratio_test, split_idx=cv_fold)
        loaders.append(iter(validloader))
        del tl, tl2

    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)
                data = data.cuda()
                label = label.cuda()

                pred = model(data)

                loss = loss_fn(pred, label)
                losses.append(loss.detach().cpu().numpy())

                _, pred = pred.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                corrects.append(correct)
                del loss, correct, pred, data, label

            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()

            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict({
                'minus_loss': -1 * np.sum(losses_min),
                'correct': np.sum(corrects_max),
                'cnt': len(corrects_max)
            })
            del corrects, corrects_max
    except StopIteration:
        pass

    del model
    metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'], elapsed_time=gpu_secs, done=True)
    return metrics['correct']


if __name__ == '__main__':
    # w = PyStopwatch()
    args = parse_search_opts()

    # if args.decay > 0:
    #     logger.info('decay=%.4f' % args.decay)
    #     C.get()['optimizer']['decay'] = args.decay

    # filehandler_name = f"{C.get()['dataset']}_{C.get()['model']['type']}_cv{args.cv_ratio}.log"
    # add_filehandler(logger, os.path.join('models', filehandler_name))
    # logger.info('configuration...')
    # logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))
    # logger.info('initialize ray...')
    # ray.init(redis_address=args.redis)

    ray.init()

    copied_c = copy.deepcopy(C.get().conf)

    # logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    # logger.info('----- Train without Augmentations cv=%d ratio (test) =%.1f -----' % (args.cv_num, args.cv_ratio))
    
    # w.start(tag='train_no_aug')
    
    saved_model_paths = [
                _get_path(C.get()['dataset'], C.get()['model']['type'], 
                        'ratio%.1f_fold%d' % (args.cv_ratio, i)) 
                for i in range(args.cv_num)
            ]
    
    # print(saved_model_paths, sep='\n')
    
    # Ray request to train base models
    requests = [
                train_model.remote(copy.deepcopy(copied_c), 
                                args.dataroot, C.get()['aug'], 
                                args.cv_ratio, i, save_path=saved_model_paths[i], skip_exist=True) 
                for i in range(args.cv_num)
            ]

    num_train_epochs = tqdm(range(C.get()['epoch']))
    is_done = False
    
    # Just checking if all models are trained for the epochs described
    # Can we try better?
    for epoch in num_train_epochs:
        while True:
            epochs_per_cv = OrderedDict()
            for cv_idx in range(args.cv_num):
                try:
                    latest_ckpt = torch.load(saved_model_paths[cv_idx])
                    if 'epoch' not in latest_ckpt:
                        epochs_per_cv['cv%d' % (cv_idx + 1)] = C.get()['epoch']
                        continue
                    epochs_per_cv['cv%d' % (cv_idx+1)] = latest_ckpt['epoch']
                except Exception as e:
                    continue
            num_train_epochs.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == args.cv_num and min(epochs_per_cv.values()) >= C.get()['epoch']:
                is_done = True
            if len(epochs_per_cv) == args.cv_num and min(epochs_per_cv.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break
    # [('wresnet40_2', 0, OrderedDict([('loss_train', 2.3426712618933783), ('loss_valid', 2.370128313700358), ('loss_test', 2.3782179090711804), ('top1_train', 0.13333333532015482), ('top1_valid', 0.11111111442248027), ('top1_test', 0.07777777893675698), ('top5_train', 0.5000000066227384), ('top5_valid', 0.4444444510671828), ('top5_test', 0.47777778572506374), ('epoch', 0)]))]

    # logger.info('getting results...')
    pretrain_results = ray.get(requests)
    print(pretrain_results)
    
    # for r_model, r_cv, r_dict in pretrain_results:
    #     logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' % (r_model, r_cv+1, r_dict['top1_train'], r_dict['top1_valid']))
    # logger.info('processed in %.4f secs' % w.pause('train_no_aug'))

    # if args.until == 1:
    #     sys.exit(0)

    # logger.info('----- Search Test-Time Augmentation Policies -----')
    # w.start(tag='search')

    # Ops = list of augmentations
    ops = augment_list(False)

    # Creating a space from which we pick data for Ray
    space = {}
    
    for i in range(args.num_policy):
        for j in range(args.num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)
    
    final_policy_set = []
    # total_computation = 0 # Just bookkeeping stuff 
    reward_attr = 'top1_valid' # top1_valid or minus_loss
    
    for _ in range(1): # run multiple times.
        for cv_fold in range(args.cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (C.get()['dataset'], C.get()['model']['type'], cv_fold, args.cv_ratio)
            register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, rpt))
            algo = HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr)

            exp_config = {
                name: {
                    'run': name,
                    'num_samples': 4 if args.smoke_test else args.num_search,
                    'resources_per_trial': {'gpu': 0},
                    'stop': {'training_iteration': args.num_policy},
                    'config': {
                        'dataroot': args.dataroot, 'save_path': saved_model_paths[cv_fold],
                        'cv_ratio_test': args.cv_ratio, 'cv_fold': cv_fold,
                        'num_op': args.num_op, 'num_policy': args.num_policy
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=args.resume, raise_on_failed_trial=False)
            
            # print()
            results = [x for x in results if x.last_result is not None]
            results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)

            # calculate computation usage
            # for result in results:
            #     total_computation += result.last_result['elapsed_time']

            for result in results[:args.num_result_per_cv]:
                final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
                # logger.info('loss=%.12f top1_valid=%.4f %s' % (result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))

                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)

    # logger.info(json.dumps(final_policy_set))
    # logger.info('final_policy=%d' % len(final_policy_set))
    # logger.info('processed in %.4f secs, gpu hours=%.4f' % (w.pause('search'), total_computation / 3600.))
    # logger.info('----- Train with Augmentations model=%s dataset=%s aug=%s ratio(test)=%.1f -----' % (C.get()['model']['type'], C.get()['dataset'], C.get()['aug'], args.cv_ratio))
    # w.start(tag='train_aug')

    num_experiments = 5
    default_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_default%d' % (args.cv_ratio, _)) for _ in range(num_experiments)]
    augment_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_augment%d' % (args.cv_ratio, _)) for _ in range(num_experiments)]
    requests = [train_model.remote(copy.deepcopy(copied_c), args.dataroot, C.get()['aug'], 0.0, 0, save_path=default_path[_], skip_exist=True) for _ in range(num_experiments)] + \
        [train_model.remote(copy.deepcopy(copied_c), args.dataroot, final_policy_set, 0.0, 0, save_path=augment_path[_]) for _ in range(num_experiments)]

    num_train_epochs = tqdm(range(C.get()['epoch']))
    is_done = False
    for epoch in num_train_epochs:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(num_experiments):
                try:
                    if os.path.exists(default_path[exp_idx]):
                        latest_ckpt = torch.load(default_path[exp_idx])
                        epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass
                try:
                    if os.path.exists(augment_path[exp_idx]):
                        latest_ckpt = torch.load(augment_path[exp_idx])
                        epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass

            num_train_epochs.set_postfix(epochs)
            if len(epochs) == num_experiments*2 and min(epochs.values()) >= C.get()['epoch']:
                is_done = True
            if len(epochs) == num_experiments*2 and min(epochs.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    final_results = ray.get(requests)

    for train_mode in ['default', 'augment']:
        avg = 0.
        for _ in range(num_experiments):
            r_model, r_cv, r_dict = final_results.pop(0)
            logger.info('[%s] top1_train=%.4f top1_test=%.4f' % (train_mode, r_dict['top1_train'], r_dict['top1_test']))
            avg += r_dict['top1_test']
        avg /= num_experiments
        logger.info('[%s] top1_test average=%.4f (#experiments=%d)' % (train_mode, avg, num_experiments))
    # logger.info('processed in %.4f secs' % w.pause('train_aug'))

    # logger.info(w)