import torch
import torch.nn as nn

from torch.nn import functional as F
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.encoder = backbone


    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, normalize=False, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            features = self.encoder(x)

            if normalize:   
                features = F.normalize(features, p=2, eps=1e-12)
            return features
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)

            if normalize:
                instance_embs = F.normalize(instance_embs, p=2, eps=1e-12)

            # print("Average norm: ", instance_embs.pow(2).sum(dim=1).mean())
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            # print(support_idx)
            # print(query_idx)
            if self.training:
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')