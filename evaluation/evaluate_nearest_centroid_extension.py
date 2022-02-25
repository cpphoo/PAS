from utils import accuracy
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np
from tqdm import tqdm

import models

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from PIL import Image
import datetime

import random


class indexed_dataset(Dataset):
    '''
    This class is a dataset class that takes in a categories file and construct a dataset
    based on the category specified in the category file 
    This is useful when we only want to evaluate certain subset of classes 
    '''

    def __init__(self, root, categories, label_space, parent_idx, transforms=None):
        '''
            label_space is a list of two items where the first is the group name of the parent
            e.g ['supercategory', 'name']
            parent idx a mapping from supercategory to index
            transform: PIL image transformations 
        '''
        super(indexed_dataset, self).__init__()
        self.transforms = transforms
        self.categories = categories
        self.root = root

        self.samples = []
        self.labels = []
        self.lvl_children = {i: [] for i in parent_idx}

        # construct samples
        for i in range(len(categories)):
            info = categories.iloc[i]
            dir_path = os.path.join(root, str(info['id']))

            # get the parent class
            parent = info[label_space[-2]]
            parent_label = parent_idx[parent]
            self.lvl_children[parent].append(i)

            imgs = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
            self.samples += imgs
            self.labels += [[parent_label, i]] * len(imgs)
        self.labels = torch.LongTensor(self.labels)

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path = self.samples[index]
        img = self._pil_loader(path)

        if transforms is None:
            return img, self.labels[index]
        else:
            return self.transforms(img), self.labels[index]

    def __len__(self):
        return len(self.samples)


def load_model(model_dict, ckpt):
    '''
        model_dict: a mapping from str to torch module
        ckpt: a string specifying the checkpoint

        load the model in model dict with checkpoint ckpt
    '''
    sd = torch.load(ckpt)

    for i in model_dict:
        model_dict[i].load_state_dict(sd[i])
    return model_dict


def generate_prototypes(reps, labels, num_categories, num_shots, 
                        reps_base, labels_base, reps_coarsely_labeled,
                        labels_coarsely_labeled, num_candidates, 
                        num_nearest_neighbors, same_parent, cosine=False):
    prototypes = []
    prototypes_label = []

    for i in range(num_categories):
        ind = torch.nonzero(labels[:, -1] == i, as_tuple=False).squeeze()

        if num_shots > 0:
            sample = torch.randperm(len(ind))[:num_shots]
        else:
            sample = torch.arange(len(ind))

        ind_l = ind[sample.cuda()]

        current_prototype = torch.mean(reps[ind_l], dim=0, keepdim=True)

        parent_identity = labels[ind_l, -2][0].item()


        # sample candidates
        if same_parent:
            ind_t = torch.nonzero(
                labels_base[:, -2] == parent_identity, as_tuple=False).squeeze()
            primary_ind = torch.randperm(len(ind_t))[:num_candidates]
            primary_ind = ind_t[primary_ind]
            ind_t = torch.nonzero(
                labels_coarsely_labeled[:, -2] == parent_identity, as_tuple=False).squeeze()
            secondary_ind = torch.randperm(len(ind_t))[:num_candidates]
            secondary_ind = ind_t[secondary_ind]
            candidates = torch.cat(
                [reps_base[primary_ind], reps_coarsely_labeled[secondary_ind]], dim=0)

        else:
            primary_ind = torch.randperm(len(reps_base))[:num_candidates]
            secondary_ind = torch.randperm(len(reps_coarsely_labeled))[
                :num_candidates]
            candidates = torch.cat(
                [reps_base[primary_ind], reps_coarsely_labeled[secondary_ind]], dim=0)

        if cosine:
            similarity = models.cosine_similarity(
                current_prototype, candidates, normalized=False)
        else:
            similarity = - \
                models.squared_l2_distance(current_prototype, candidates)

        similarity = similarity.squeeze()
        nearest_neigbhors = torch.argsort(similarity, dim=0, descending=True)[
            :num_nearest_neighbors]

        refined_prototypes = torch.mean(torch.cat([reps[ind_l], 
                            candidates[nearest_neigbhors]], dim=0),
                                        dim=0)

        prototypes.append(refined_prototypes)

        prototypes_label.append(i)

    prototypes = torch.stack(prototypes)

    # need to normalize the prototypes again after addition
    if cosine:
        prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)

    prototypes_label = torch.LongTensor(prototypes_label)

    return prototypes, prototypes_label


def generate_logits(query, centroids, cosine=False):
    '''
        Helper function to generate logits
    '''
    if cosine:
        metric = 'cosine_with_normalized_features'
    else:
        metric = 'squared_l2'

    logits = models.predict(centroids=centroids, query=query, metric=metric)
    return logits


def generate_reps(backbone, dloader, cosine=False):
    '''
        Do a forward pass on backbone on all the data in dloader
    '''
    reps = []
    labels = []

    for X, y in tqdm(dloader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.no_grad():
            rep = backbone(X)
        reps.append(rep)
        labels.append(y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)

    if cosine:
        reps = F.normalize(reps, dim=1, p=2, eps=1e-12)
    return reps, labels


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Need to create dataloader
    # load indices for novel categories
    # base_categories = pd.read_csv(args.base_categories)
    novel_categories = pd.read_csv(args.novel_categories)
    parent_names = list(
        np.sort(np.unique(novel_categories[args.label_space[-2]])))
    num_parent = len(parent_names)

    parent_idx = {i: ind for ind, i in enumerate(parent_names)}

    num_novel = len(novel_categories)

    if args.image_transformations == 'standard':
        transforms_final = transforms.Compose([
            transforms.CenterCrop(84),
            transforms.ToTensor(),
        ])
    elif args.image_transformations == 'standard_cifar':
        transforms_final = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Invalid Image Transformation!')

    # Construct the base and coarsely labeled set
    base_set = indexed_dataset(args.base_path, 
                                pd.read_csv(args.base_label_file), 
                                args.base_label_space,
                                parent_idx, transforms_final)

    coarsely_labeled_set = indexed_dataset(args.coarsely_labeled_path, 
                                           pd.read_csv(
                                               args.coarsely_labeled_label_file),
                                           args.coarsely_labeled_label_space,
                                         parent_idx, transforms_final)

    base_dloader = DataLoader(base_set, batch_size=args.bsize, 
                            num_workers=args.num_workers,
                              shuffle=False, drop_last=False, 
                              pin_memory=True)

    coarsely_labeled_dloader = DataLoader(coarsely_labeled_set,
                                        batch_size=args.bsize, 
                                        num_workers=args.num_workers,
                                        shuffle=False, drop_last=False, pin_memory=True)

    # Construct a reference and query set
    ref_novel_dset = indexed_dataset(
        args.ref_path, novel_categories, args.label_space, parent_idx,
        transforms_final)

    query_novel_dset = indexed_dataset(
        args.query_path, novel_categories, args.label_space, parent_idx,
        transforms_final)

    ref_novel_dloader = DataLoader(ref_novel_dset,
                                   batch_size=args.bsize,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   drop_last=False, pin_memory=True)

    query_novel_dloader = DataLoader(query_novel_dset,
                                     batch_size=args.bsize,
                                     num_workers=args.num_workers,
                                     shuffle=False,
                                     drop_last=False, pin_memory=True)

    # Create model
    model_dict = {}
    backbone = models.resnet18(
        remove_last_relu=args.remove_last_relu, high_res_input=False)
    # load the model
    model_dict['backbone'] = backbone.cuda()

    # detach the model
    for i in model_dict:
        for p in model_dict[i].parameters():
            p.requires_grad_(False)

    model_dict = load_model(model_dict, args.ckpt)
    model = model_dict['backbone'].eval()

    # To keep track of the performance
    result_average = [
        {str(s): np.zeros(args.num_repetitions) for s in args.shots}
    ]

    result_average += [
        {**{str(s): np.zeros(args.num_repetitions) for s in args.shots},
         **{str(s) + '_restricted': np.zeros(args.num_repetitions) for s in args.shots}}
        for _ in range(num_parent)
    ]

    result_per_class = [
        {str(s): np.zeros(args.num_repetitions) for s in args.shots}
    ]

    result_per_class += [
        {**{str(s): np.zeros(args.num_repetitions) for s in args.shots},
         **{str(s) + '_restricted': np.zeros(args.num_repetitions) for s in args.shots}}
        for _ in range(num_parent)
    ]

    # name for each tab
    # naming: all and parent_classes
    names = ['novel_all'] + ['novel_' + p for p in parent_names]

    reps_base, labels_base = generate_reps(model, base_dloader, args.cosine)
    reps_coarsely_labeled, labels_coarsely_labeled = generate_reps(model, 
                                                coarsely_labeled_dloader,
                                                    args.cosine)

    # generate embeddings for the query examples
    reps_ref_novel, labels_ref_novel = generate_reps(
        model, ref_novel_dloader, args.cosine)

    # # This might take up a lot of the gpu memory
    # # Move this to cpu for now and then when prototypes are generated,
    # reps_ref_novel = reps_ref_novel.cpu()

    # generate embeddings for the query examples
    reps_query_novel, labels_query_novel = generate_reps(
        model, query_novel_dloader, args.cosine)

    for s in args.shots:
        if s > 0:
            num_repetitions = args.num_repetitions
        else:
            # for nonpositive shot, we use all reference examples
            # in that case, we only need to evaluate once
            num_repetitions = 1

        for i in tqdm(range(num_repetitions)):
            # generate prototypes
            prototypes_novel, _ = generate_prototypes(
                reps_ref_novel, labels_ref_novel, num_novel, s, 
                reps_base, labels_base,
                reps_coarsely_labeled,
                labels_coarsely_labeled, num_candidates=args.num_candidates,
                num_nearest_neighbors=args.num_nearest_neighbors,
                same_parent=args.same_parent,
                cosine=args.cosine)

            prototypes_novel = prototypes_novel.cuda()

            # Evaluate on query
            logits_novel = generate_logits(
                reps_query_novel, prototypes_novel, args.cosine)

            # compute the accuracy
            t = accuracy(logits_novel, labels_query_novel[:, -1])

            result_average[0][str(s)][i] = t['average'][0].item()
            result_per_class[0][str(s)][i] = t['per_class_average'][0].item()

            # Compute the accuracy for individual parent
            for p in parent_names:
                parent_id = parent_idx[p]

                ind = torch.nonzero(
                    query_novel_dset.labels[:, -2] == parent_id, as_tuple=False).squeeze()
                target_label = labels_query_novel[ind, -1]

                # calculate the accuracy
                t = accuracy(logits_novel[ind], target_label)
                result_average[parent_id +
                               1][str(s)][i] = t['average'][0].item()
                result_per_class[parent_id +
                                 1][str(s)][i] = t['per_class_average'][0].item()

                # calculate the accuracy among siblings
                children_id = query_novel_dset.lvl_children[p]

                # need to relabel the labels since the function only consider
                # [num_classes]

                labels_query_novel_relabel = torch.zeros_like(target_label)

                for ind_x, x in enumerate(children_id):
                    ind_temp = torch.nonzero(target_label == x, as_tuple=False)
                    labels_query_novel_relabel[ind_temp] = ind_x

                # calculate the accuracy for the unrestricted space
                t = accuracy(
                    logits_novel[ind.view(-1, 1), children_id], labels_query_novel_relabel)
                result_average[parent_id +
                               1][str(s) + '_restricted'][i] = t['average'][0].item()
                result_per_class[parent_id + 1][str(
                    s) + '_restricted'][i] = t['per_class_average'][0].item()

        if s <= 0:
            # repeat all shots so that it is consistent with other number of shots
            result_average[0][str(s)] = result_average[0][str(s)][0]
            result_per_class[0][str(s)] = result_per_class[0][str(s)][0]

            for i in range(1, num_parent + 1):
                result_average[i][str(s)] = result_average[i][str(s)][0]
                result_per_class[i][str(s)] = result_per_class[i][str(s)][0]

                result_average[i][str(
                    s) + '_restricted'] = result_average[i][str(s) + '_restricted'][0]
                result_per_class[i][str(
                    s) + '_restricted'] = result_per_class[i][str(s) + '_restricted'][0]

        # construct the filename
        if args.result_file is None:
            result_file = "result_" + str(datetime.datetime.now())
        else:
            result_file = args.result_file

        # write the results
        with pd.ExcelWriter(result_file + '.xlsx', engine='xlsxwriter') as f:
            for ind, i in enumerate(names):
                pd.DataFrame(result_per_class[ind]).to_excel(
                    f, sheet_name=('per_class_' + i)[:31])

            for ind, i in enumerate(names):
                pd.DataFrame(result_average[ind]).to_excel(
                    f, sheet_name=('average_' + i)[:31])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--ref_path', type=str, required=True,
                        help='path to all reference images')
    parser.add_argument('--query_path', type=str,
                        required=True, help='path to all query images')
    parser.add_argument('--label_space', type=str, nargs='+', default=['supercategory', 'name'],
                        help='label space to consider')
    parser.add_argument('--novel_categories', type=str, required=True,
                        help='path to a csv file that has the novel categories')
    parser.add_argument('--image_transformations', type=str,
                        default=None, help='Image Transformations to use')
    parser.add_argument('--base_path', type=str,
                        required=True, help='path to the base datasets')
    parser.add_argument('--base_label_space', type=str, nargs='+', default=['supercategory', 'name'],
                        help='label space to consider for base dataset')
    parser.add_argument('--base_label_file', type=str,
                        help='a csv file containing all the label for the base dataset to be considered')
    parser.add_argument('--coarsely_labeled_path', type=str,
                        required=True, help='directory for the coarsely-labeled dataset')
    parser.add_argument('--coarsely_labeled_label_space', type=str, nargs='+', default=['supercategory', 'name'],
                        help='label space to consider for the coarsely-labeled dataset')
    parser.add_argument('--coarsely_labeled_label_file', type=str,
                        help='a csv file containing all the label to be considered for the coarsely-labeled dataset')
    
    
    parser.add_argument('--bsize', type=int, default=512,
                        help='batch_size for loading data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of threads for dataloading')

    # models
    parser.add_argument('--ckpt', type=str, required=True,
                        help='model to evaluate')
    parser.add_argument('--cosine', action='store_true',
                        help='use cosine similarity')
    parser.add_argument('--remove_last_relu', action='store_true',
                        help='whether to remove the last relu activation for resnet')

    # bookkeeping
    parser.add_argument('--result_file', type=str,
                        help='Filename for the result')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')

    # Task specification
    parser.add_argument('--num_repetitions', type=int, default=10,
                        help='num of repeats')
    parser.add_argument('--shots', type=int, nargs='+', default=[0, 1, 2, 5, 10],
                        help='the number of shots')
    parser.add_argument('--num_nearest_neighbors', type=int, default=10,
                        help='num of neighbors for prototype refinement')
    parser.add_argument('--num_candidates', type=int, default=100,
                        help='num of candidates for prototype refinement')
    parser.add_argument('--same_parent', action='store_true',
                        help='use same parent when picking candidates')
    args = parser.parse_args()
    main(args)
