import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from tqdm import tqdm
import argparse

import os
import numpy as np
import random

import utils

import models
import data
import time

import wandb
import warnings

import copy
import math

def main(args):
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, 'checkpoint.log'), __name__)
    trainlog = utils.savelog(args.dir, 'train')

    wandb.init(project='few_shot_coarse_labels',
               group=__file__,
               name=f'{__file__}_{args.dir}')

    wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create DataLoader
    ###########################
    if args.image_transformations == 'standard':
        transforms_final = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transforms_test = transforms.Compose([
            transforms.CenterCrop(84),
            transforms.ToTensor(),
        ])
    elif args.image_transformations == 'standard_cifar':
        transforms_final = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transforms_test = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Invalid Image Transformation!')


    trainset = data.ImageFolder(root=args.trainpath,
                                label_file=args.label_file,
                                label_space=args.label_space,
                                transform=transforms_final)
    ############################

    ###########################
    # Create Models
    ###########################
    backbone = models.resnet18(
        remove_last_relu=args.remove_last_relu, high_res_input=False)
    feature_dim = 512

    backbone = nn.DataParallel(backbone).cuda()
    num_base_classes = len(trainset.class_hierarchy[-1])
    
    if args.use_cosine_clf:
        if not args.remove_last_relu:
            warnings.warn(
                "Using cosine classifier without the last relu activation removed!")
        clf = models.cosine_clf(feature_dim, num_base_classes).cuda()
        
    else:
        if args.remove_last_relu:
            warnings.warn(
                "Using linear classifier with the last relu activation removed!")
        clf = nn.Linear(feature_dim, num_base_classes).cuda()
    ############################

    ############################################
    ### Create the dataset for self-training
    ############################################

    sd_clf_init = copy.deepcopy(clf.state_dict())
    sd_backbone_init = copy.deepcopy(backbone.state_dict())

    # load the teacher model
    teacher_state_dict = torch.load(args.teacher_state_dict)
    backbone.module.load_state_dict(teacher_state_dict['backbone'])
    clf.load_state_dict(teacher_state_dict['clf'])

    # this is the coarsely labeled data
    secondary_trainset = data.ImageFolder(args.secondary_trainpath, 
                                          label_file=args.secondary_label_file,
                                          label_space=args.secondary_label_space,
                                          transform=transforms_final,
                                          base_taxonomy=trainset.class_hierarchy)

    secondary_trainset = relabel_dataset(
        secondary_trainset, args.coarse_label_ratio)
    relabeling_ratio = (
        secondary_trainset.targets[:, -2] == -1).float().sum() / len(secondary_trainset)
    print("Unlabeled ratio: ", relabeling_ratio.item())

    logger.info(
        f'Ratio of non labeled coarsely labeled examples: {relabeling_ratio.item()}')

    torch.save({
        "samples": secondary_trainset.samples,
        "targets": secondary_trainset.targets,
        "class_hierarchy": secondary_trainset.class_hierarchy,
        "class_to_idx_hierarchy": secondary_trainset.class_to_idx_hierarchy
    },
        os.path.join(args.dir, 'coarse_dataset.pkl'))

    # make sure the coarse labels ordering is the same
    assert trainset.class_hierarchy[-2] == secondary_trainset.class_hierarchy[-2]

    secondary_trainset = generate_soft_pseudolabels(backbone, clf, trainset, secondary_trainset, 
                            transforms_final, transforms_test, args)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.bsize, 
                                              shuffle=True, 
                                              drop_last=True,
                                              num_workers=args.num_workers, 
                                              pin_memory=True)

    secondary_trainloader = torch.utils.data.DataLoader(secondary_trainset,
                                                        batch_size=args.secondary_bsize,
                                                        shuffle=True, drop_last=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    num_coarse_classes = len(trainset.class_hierarchy[-2])
    num_novel_seen_classes = len(secondary_trainset.class_hierarchy[-1])

    logger.info(f"num_classes: {num_base_classes}")
    logger.info(f"num_coarse_classes: {num_coarse_classes}")
    logger.info(f"num_novel_seen_classes: {num_novel_seen_classes}")
    logger.info(
        f"num_iterations_per_epoch: {len(trainloader)}")
    logger.info(
        f"num_iterations_full_pass_coarse_data: {len(secondary_trainloader)}")

    backbone.load_state_dict(sd_backbone_init)
    clf.load_state_dict(sd_clf_init)
    ###########################
    # Create Optimizer
    ###########################

    optimizer = torch.optim.SGD([
        {'params': backbone.parameters()},
        {'params': clf.parameters()}
    ],
        lr=args.lr, momentum=0.9,
        weight_decay=args.wd,
        nesterov=False)
    ############################

    ############################
    # Specify Loss Function
    ############################
    criterion = nn.KLDivLoss(reduction='batchmean')
    ############################
    starting_epoch = 0

    if args.load_path is not None:
        print('Loading model from {}'.format(args.load_path))
        logger.info('Loading model from {}'.format(args.load_path))
        starting_epoch = load_checkpoint(
            backbone, clf, optimizer, args.load_path)

    if args.resume_latest:
        # Only works if model is saved as checkpoint_(/d+).pkl
        import re
        pattern = "checkpoint_(\d+).pkl"
        candidate = []
        for i in os.listdir(args.dir):
            match = re.search(pattern, i)
            if match:
                candidate.append(int(match.group(1)))

        # if nothing found, then start from scratch
        if len(candidate) == 0:
            print('No latest candidate found to resume!')
            logger.info('No latest candidate found to resume!')
        else:
            latest = np.amax(candidate)
            load_path = os.path.join(args.dir, f'checkpoint_{latest}.pkl')
            if latest >= args.epochs:
                print('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                logger.info('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                import sys
                sys.exit(0)
            else:
                print('Resuming from the latest checkpoint: {}'.format(load_path))
                logger.info(
                    'Resuming from the latest checkpoint: {}'.format(load_path))
                if args.load_path:
                    logger.info(
                        'Overwriting model loaded from {}'.format(args.load_path))
                starting_epoch = load_checkpoint(
                    backbone, clf, optimizer, load_path)

    # save the initialization
    checkpoint(backbone, clf, optimizer, trainset.class_hierarchy, 
                trainset.class_to_idx_hierarchy, 
    os.path.join(
        args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch)

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):
            train(backbone, clf, optimizer, trainloader, secondary_trainloader, 
            criterion, epoch, args.epochs, logger, trainlog, args)

            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(backbone, clf, optimizer,  trainset.class_hierarchy, 
                trainset.class_to_idx_hierarchy,
                os.path.join(
                    args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(backbone, clf, optimizer,  trainset.class_hierarchy,
                       trainset.class_to_idx_hierarchy, os.path.join(
                args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)
    finally:
        trainlog.save()
    return


def checkpoint(backbone, clf, optimizer, class_hierarchy, class_to_idx_hierarchy, 
            save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    torch.save({
        'backbone': backbone.module.state_dict(),
        'clf': clf.state_dict(),
        'opt': optimizer.state_dict(),
        'class_hierarchy': class_hierarchy,
        'class_to_idx_hierarchy': class_to_idx_hierarchy, 
        'epoch': epoch
    }, save_path)
    return


def load_checkpoint(backbone, clf, optimizer, load_path):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path)
    backbone.module.load_state_dict(sd['backbone'])
    clf.load_state_dict(sd['clf'])
    optimizer.load_state_dict(sd['opt'])
    return sd['epoch']


def lr_schedule_step(optimizer, epoch, step_in_epoch, total_steps_in_epochs, args):
    '''
        Warm up the learning rate by 1 epoch

        Drops the learning rate by a factor of 10 every one-third of the total number
        of training epochs
    '''
    step = epoch + step_in_epoch / total_steps_in_epochs

    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length

    #### linear rampup for the first epochs
    lr = linear_rampup(step, 1) * args.lr

    #### dropdown the learning rate by a factor of 10 every lr decay freq epochs
    def step_rampdown(epoch):
        return 0.1 ** (epoch // args.lr_decay_freq)

    lr = step_rampdown(epoch) * lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def relabel_dataset(dataset, coarse_label_ratio):
    '''
        For each fine-grained class, keep the ratio of the coarse-labeled examples to be 
        coarse_label_ratio
    '''
    assert (coarse_label_ratio >= 0 and coarse_label_ratio <= 1)

    if coarse_label_ratio == 1:
        return dataset

    targets = dataset.targets
    num_fine_grained_classes = len(dataset.class_hierarchy[-1])

    for i in range(num_fine_grained_classes):
        ind = torch.nonzero(targets[:, -1] == i, as_tuple=False).squeeze()

        num_examples = len(ind)
        cutoff = math.ceil(num_examples * coarse_label_ratio)
        ind_to_be_relabel = ind[torch.randperm(num_examples)[cutoff:]]

        targets[ind_to_be_relabel, -2] = -1

    dataset.targets = targets
    # print("After reLabeling: ", targets[:10, -2])

    dataset.imgs = [[dataset.samples[i], dataset.targets[i]]
                    for i in range(len(dataset.samples))]
    return dataset


def parent_consistent_filtering(y_parent, logits_pred, lvl_child, class_to_idx_hierarchy,
                                class_hierarchy):
    num_base_classes = len(class_hierarchy[-1])

    soft_pseudolabels = torch.zeros((len(y_parent), num_base_classes))

    num_parent_classes = len(lvl_child[-1])

    for i in range(num_parent_classes):
        # get the indices of examples with parent label i
        idx = torch.nonzero(y_parent == i, as_tuple=False)

        if len(idx) == 0:
            continue

        parent_name = class_hierarchy[-2][i]

        # lvl_child[-1] is a mapping from parent to children classes
        # construct the children idx 
        children_idx = [class_to_idx_hierarchy[-1][j] for j in lvl_child[-1][parent_name]]

        soft_pseudolabels[idx, children_idx] = F.softmax(logits_pred[idx, children_idx], dim=1)

        assert torch.isclose(soft_pseudolabels[idx.squeeze()].sum(
            dim=-1).mean(), torch.ones(1))

    # handle the unknown parent class
    idx = torch.nonzero(y_parent == -1, as_tuple=False).squeeze()

    if len(idx) != 0:
        print("Handling Unknown parent! No filtering is done")
        soft_pseudolabels[idx] = F.softmax(logits_pred[idx], dim=1)
        assert torch.isclose(soft_pseudolabels[idx.squeeze()].sum(
            dim=-1).mean(), torch.ones(1))

    return soft_pseudolabels

@torch.no_grad()
def generate_soft_pseudolabels(backbone_teacher, clf_teacher, primary_trainset, secondary_trainset,
            transforms_train, transforms_test, args):
    '''
        Generate the pseudolabels needed for self-training

        primary dataset - the dataset that is labeled
        secondary_trainset - the dataset to be pseudolabeled
    '''
    backbone_teacher.eval()
    clf_teacher.eval()

    # change the transformation for the secondary trainset to deterministic transform
    secondary_trainset.transform = transforms_test

    loader = torch.utils.data.DataLoader(secondary_trainset,
                                         batch_size=args.bsize,
                                         shuffle=False, drop_last=False,
                                         num_workers=args.num_workers,
                                         pin_memory=False)

    logits = []
    ys = []

    for (X, y) in loader:
        X = X.cuda(non_blocking=True)
        
        features = backbone_teacher(X)
        logit = clf_teacher(features)

        logits.append(logit)
        ys.append(y)

    logits = torch.cat(logits, dim=0).cpu()
    ys = torch.cat(ys, dim=0)

    # Do parent consistent filtering
    if args.parent_consistent: 
        soft_pseudolabels = parent_consistent_filtering(ys[:, -2], logits, primary_trainset.lvl_child, 
                                    primary_trainset.class_to_idx_hierarchy,
                                    primary_trainset.class_hierarchy)
    else:
        soft_pseudolabels = F.softmax(logits, dim=1)

    # construct the new targets
    # the new labels are represented as soft_pseudolabels, ground truth
    new_targets = torch.cat([soft_pseudolabels, ys.float()], dim=1)
    secondary_trainset.targets = new_targets
    secondary_trainset.imgs = [[secondary_trainset.samples[i], secondary_trainset.targets[i]]
                               for i in range(len(secondary_trainset.samples))]
    
    # change the transformation for the secondary trainset back to stochastic transform
    secondary_trainset.transform = transforms_train

    backbone_teacher.train()
    clf_teacher.train()
    return secondary_trainset

def train(backbone, clf, optimizer, trainloader, secondary_trainloader, criterion, 
                epoch, num_epochs, logger, trainlog, args):
    '''
        Train the classifier (backbone, clf) by one epoch
    '''
    meters = utils.AverageMeterSet()
    backbone.train()
    clf.train()

    num_base_classes = len(trainloader.dataset.class_hierarchy[-1])

    secondary_trainloader_iter = iter(secondary_trainloader)

    start_data = torch.cuda.Event(enable_timing=True)
    end_data = torch.cuda.Event(enable_timing=True)

    start_update = torch.cuda.Event(enable_timing=True)
    end_update = torch.cuda.Event(enable_timing=True)

    end = time.perf_counter()
    torch.cuda.synchronize()
    start_data.record()
    for i, (X, y) in enumerate(trainloader):
        n_fine_grained = len(X)
        try:
            X_secondary, y_secondary = secondary_trainloader_iter.next()
        except StopIteration:
            secondary_trainloader_iter = iter(secondary_trainloader)
            X_secondary, y_secondary = secondary_trainloader_iter.next()

        y = y.cuda(non_blocking=True)
        y_one_hot = F.one_hot(y[:, -1], num_base_classes).float()
        y_secondary_pseudolabel = y_secondary[:, :num_base_classes].cuda(non_blocking=True)

        X_combined = torch.cat([X.cuda(non_blocking=True), 
                                X_secondary.cuda(non_blocking=True)], 
                                dim=0)
        end_data.record()

        start_update.record()
        lr = lr_schedule_step(optimizer, epoch, i, len(trainloader), args)
        
        optimizer.zero_grad()
        features = backbone(X_combined)
        logits = clf(features)

        perf = utils.accuracy(logits[:n_fine_grained].data, y[:, -1].data, topk=(1, 5), 
                    compute_mean_class=False)

        log_probability = F.log_softmax(logits, dim=1)
        loss_fine_grained = criterion(log_probability[:n_fine_grained], y_one_hot)
        loss_self_training = criterion(log_probability[n_fine_grained:], y_secondary_pseudolabel)

        loss = loss_fine_grained + loss_self_training

        loss.backward()
        optimizer.step()

        end_update.record()

        torch.cuda.synchronize()
        meters.update('Iteration_time', time.perf_counter() - end, 1)
        meters.update('Data_time', start_data.elapsed_time(end_data) * 1e-3, 1)
        meters.update('Update_time', start_update.elapsed_time(end_update) * 1e-3, 1)
        meters.update('Loss', loss.item(), 1)
        meters.update('Loss_supervised_fine_grained', loss_fine_grained.item(), 1)
        meters.update('Loss_self_training', loss_self_training.item(), 1)
        meters.update('Lr', lr, 1)
        meters.update('top1_fine_grained', perf['average'][0].item(), 1)
        meters.update('top5_fine_grained', perf['average'][1].item(), 1)

        if (i + 1) % args.print_freq == 0:
            logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}] '
                                '{meters:.5f} ').format(
                epoch=epoch, epochs=num_epochs, step=i+1, 
                steps=len(trainloader), meters=meters)

            logger.info(logger_string)

        if (args.iteration_bp is not None) and (i+1) == args.iteration_bp:
            break

        end = time.perf_counter()
        start_data.record()

    logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step}] '
                    '{meters:.5f} ').format(
                    epoch=epoch+1, epochs=num_epochs, step=0, 
                    meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    trainlog.record(epoch, {
        **values,
        **averages,
        **sums
    })

    wandb.log({'loss': averages['Loss/avg'],
                'loss_supervised_fine_grained': averages['Loss_supervised_fine_grained/avg'],
                'loss_self_training': averages['Loss_self_training/avg']}, step=epoch+1)
    wandb.log({
        'top1_fine_grained': averages['top1_fine_grained/avg'],
        'top5_fine_grained': averages['top5_fine_grained/avg'],
              }, step=epoch+1)

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to train the Self-training Representation')
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to save the checkpoints')

    # Method specific
    parser.add_argument('--teacher_state_dict', type=str,
                        required=True, help='filepath to teacher state dict')
    parser.add_argument('--parent_consistent', action='store_true',
                        help='whether to do parent consistent filtering')
    parser.add_argument('--coarse_label_ratio', type=float, default=1, 
                        help='Amount of coarse label ratio per fine-grained class. (This argument is for ablation analysis)')


    # dataset and dataloaders
    parser.add_argument('--trainpath', type=str,
                        default='~/data', help='directory to look for the training data')
    parser.add_argument('--label_space', type=str, nargs='+', 
                        default=['supercategory', 'name'], 
                        help='label space to consider')
    parser.add_argument('--label_file', type=str,
                        help='a csv file containing all the label to be considered')
    parser.add_argument('--secondary_trainpath', type=str,
                        default='~/data', help='directory to look for the coarsely-labeled data')
    parser.add_argument('--secondary_label_space', type=str, nargs='+',
                        default=['supercategory', 'name'],
                        help='label space to consider for the coarsely-labeled data')
    parser.add_argument('--secondary_label_file', type=str,
                        help='a csv file containing all the label to be considered for coarsely-labeled data')
    parser.add_argument('--image_transformations', type=str, 
                        default=None, help='Image Transformations to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')

    # model
    parser.add_argument('--remove_last_relu', action='store_true',
                        help='whether to remove the last relu activation for resnet')
    parser.add_argument('--use_cosine_clf', action='store_true',
                        help='whether to use cosine classifier')


    # optimization
    parser.add_argument('--bsize', type=int, default=256,
                        help='batch_size for labeled data')
    parser.add_argument('--secondary_bsize', type=int, default=256,
                        help='batch_size for coarsely-labeled data')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (one epoch is a complete pass of the labeled data)')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate for model')
    parser.add_argument('--lr_decay_freq', type=int, default=100,
                        help='learning rate decay frequency (in epochs)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='Weight decay for the model')


    # miscellany 
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--iteration_bp', type=int,
                        help='number of iteration to break the training loop. Useful for debugging')
    args = parser.parse_args()
    main(args)
