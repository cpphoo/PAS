# Compute accuracy
import torch

@torch.no_grad()
def accuracy(logits, ground_truth, topk=[1, ], compute_mean_class=True):
    '''
        Compute the topk accuracy. 

        logits:  n, k matrix
        ground truth: vector of len k
        topk: a list of topk results to compute
        compute_mean_class: whether to compute average per class accuracy 
    '''
    assert len(logits) == len(ground_truth)

    n, d = logits.shape

    acc = {}
    acc['topk'] = topk
    acc['num_classes'] = d
    acc['average'] = []

    max_k = max(topk)
    _, pred = logits.topk(k=max_k, dim=1, largest=True, sorted=True)

    # argsort = torch.argsort(logits, dim=1, descending=True)[:, :min([max_k, d])]
    correct = (pred == ground_truth.view(-1, 1)).float()

    for indj, j in enumerate(topk):
        num_correct = torch.sum(correct[:, :j])
        acc['average'].append(num_correct / n * 100)

    if compute_mean_class:
        label_unique = torch.unique(ground_truth)
        acc['per_class_average'] = torch.zeros(len(topk)).to(correct.device)
        acc['per_class'] = [[] for _ in label_unique]
        acc['gt_unique'] = label_unique

        for indi, i in enumerate(label_unique):
            ind = torch.nonzero(ground_truth == i).view(-1)
            correct_target = correct[ind]

            # calculate topk
            for indj, j in enumerate(topk):
                num_correct_partial = torch.sum(correct_target[:, :j])
                acc_partial = num_correct_partial / len(correct_target)
                acc['per_class_average'][indj] += acc_partial
                acc['per_class'][indi].append(acc_partial * 100)

        acc['per_class_average'] = acc['per_class_average'] / \
            len(label_unique) * 100

    return acc


@torch.no_grad()
def top1_accuracy(logits, ground_truth):
    '''
        Computes the top1 accuracy
    '''
    assert len(logits) == len(ground_truth)

    pred = logits.argmax(dim=1)
    acc = (pred == ground_truth).float().mean() * 100
    return acc
