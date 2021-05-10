import torch
from statistics import median

def compute_mr(query, gallery, target_query, target_gallery, k, reverse=False):
    result = []
    query = query / query.norm(dim=1,keepdim=True)
    gallery = gallery / gallery.norm(dim=1,keepdim=True)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(mr(sim_cosine, target_gallery, target_query, k))
    if reverse:
        result.extend(mr(sim_cosine, target_query, target_gallery, k, dim=0))
    return result


def mr(sim, target_gallery, target_query, k, dim=1):
    result = []
    maxk = k
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    ranks = []
    for row in correct.t():
        temp = torch.where(row > 0)[0]
        if temp.shape[0] > 0:
            ranks.append(temp[0].item()  + 1)
        else:
            ranks.append(k)
            print('incr. k')

    result.append(median(ranks) * 100 / size_total)
    return result