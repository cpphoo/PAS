import torch 
from torch.nn import functional as F

def squared_l2_distance(A, B):
    '''
        Compute the squared l2 distance between two feature matrices

        A: n by d feature matrix
        B: m by d feature matrix 

        Output: n x m matrix where the (i, j)-th entry is the squared l2 
        distance between A[i] and B[j]
    '''
    norm_A = torch.sum(A ** 2, dim=1).view(-1, 1)
    norm_B = torch.sum(B ** 2, dim=1).view(1, -1)

    AB = torch.matmul(A, B.t())
    square_dist = torch.clamp(norm_A + norm_B - 2*AB, min=0)
    return square_dist

def cosine_similarity(A, B, normalized=True):
    '''
        Compute the cosine similarity between two feature matrices

        A: n by d feature matrix
        B: m by d feature matrix 
        normalized: whether the each row of A and B are normalized to unit norm

        Output: n x m matrix where the (i, j)-th entry is the cosine similarity
        between A[i] and B[j]
    '''
    if normalized:
        A_normalized = A
        B_normalized = B
    else:
        A_normalized = F.normalize(A, p=2, dim=1, eps=1e-12)
        B_normalized = F.normalize(B, p=2, dim=1, eps=1e-12)
    return torch.matmul(A_normalized, B_normalized.t())


def predict(centroids, query, metric='squared_l2'):
    '''
        A simple wrapper functions to computes the similarities between two 
        feature matrices centroids and query
    '''
    if metric == 'squared_l2':
        similarity = -squared_l2_distance(query, centroids)
    elif metric == 'cosine_with_normalized_features':
        similarity = cosine_similarity(query, centroids, normalized=True)
    elif metric == 'cosine_with_unnormalized_features':
        similarity = cosine_similarity(query, centroids, normalized=False)
    else:
        raise ValueError('Invalid metric specified')
    return similarity
