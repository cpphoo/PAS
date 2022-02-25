import torch
import torch.nn as nn
from .metric import predict

class cosine_clf(nn.Module):
    '''
        A classifier based on cosine similarities with a learnable temperature parameter 
    '''
    def __init__(self, in_features, out_features, temperature=10):
        super(cosine_clf, self).__init__()
        self.W = nn.Parameter(torch.randn((out_features, in_features)), requires_grad=True)
        self.temp = nn.Parameter(torch.FloatTensor([temperature]), requires_grad=True)


    def forward(self, x):
        assert len(x.size()) == 2
        similarity = predict(self.W, x, metric='cosine_with_unnormalized_features')

        return self.temp * similarity