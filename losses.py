import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def get_loss(name):
    if name == "NCE":
        return InfoNCE_default
    elif name == "pos-out-single":
        return InfoNCE_pos_out_single


def InfoNCE_default(batch_size, n_views, temperature, features, device):
    # Here args.n_views = 2
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    # For each subject, it has (n_views-1) positive and (batch_size-1)*n_views negatives
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

    # labels: (batch_size, batch_size-1)
    labels = labels[~mask].view(labels.shape[0], -1)
    # similarity_matrix: (batch_size, batch_size-1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives
    # positives: (batch_size, n_views-1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select negatives
    # negatives: (batch_size, batch_size-n_views)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Now for each subject, its positive is the first column
    logits = torch.cat([positives, negatives], dim=1).to(device)
    # So the ground truth label should all be zero
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature

    criteration = torch.nn.CrossEntropyLoss().to(device)
    loss = criteration(logits, labels)

    return loss


def InfoNCE_pos_out_single(batch_size, n_views, temperature, features, device):
    """
    Each time we only consider one pos pair in numerator, and one pos pair + all neg pairs in denominator
    :param batch_size:
    :param n_views:
    :param temperature:
    :param features:
    :param device:
    :return:
    """
    features = F.normalize(features.type(torch.float32), dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # labels: (batch_size * n_views, batch_size * n_views)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    loss_total = 0.0

    # Each time, for each sample, its postive is different, but negatives are constant. That means we only need to select
    # one positive pair each time and mask the others.
    for n in range(0, n_views-1):
        # Suppose we first mask all positive pairs, then we only need to recover the selected pos-pairs according to n.
        mask = torch.eye(batch_size, dtype=bool).repeat(n_views, n_views)
        # Below is not a 2-layer loop, just for ease of traversal.
        for i in range(0, batch_size):
            for j in range(0, n_views):
                row = i + j * batch_size
                # the i-th pos for each anchor is its i-th following samples augmented from the same ancestor
                # Example: We augment A for three times and get A1, A2 and A3, which means n_views = 3. For A1, its 1st
                #          pos is A2, 2nd is A3, for A2, its 1st pos is A3, 2nd is A1, for A3, its 1st pos is A1,
                #          2nd pos is A2.
                column = i + ((j+n+1) % n_views) * batch_size
                mask[row][column] = False
        mask = ~mask
        # labels_temp: (batch_size*n_views, 1 + (batch_size-1) * n_views)
        labels_temp = labels[mask].bool().view(batch_size*n_views, -1)
        # similarity_matrix_temp: (batch_size*n_views, 1 + (batch_size-1) * n_views)
        similarity_matrix_temp = similarity_matrix[mask].view(batch_size*n_views, -1)

        pos = similarity_matrix_temp[labels_temp.bool()].view(similarity_matrix_temp.size()[0], -1) / temperature
        neg = similarity_matrix_temp[~labels_temp.bool()].view(similarity_matrix_temp.shape[0], -1) / temperature

        logits = torch.cat([pos, neg], dim=1).to(device)
        logits = F.softmax(logits, dim=1)

        pos = logits[:, 0]
        loss_total += (-torch.log(pos).mean())

    # calculate the average outside the log
    loss = loss_total / (n_views - 1)

    return loss


class LossFun(nn.Module):
    def __init__(self, configs, train_batch, distance_type):
        super(LossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        self.triplets_dis = np.load(configs.path_triplets_truth)

    def forward(self, embedding_a, embedding_p, embedding_n, batch_index):
        batch_triplet_dis = torch.tensor(self.triplets_dis[batch_index], dtype=torch.float32).cuda()
        true = torch.concat([batch_triplet_dis[:, 0].unsqueeze(-1), batch_triplet_dis[:, 1].unsqueeze(-1)], dim=1)
        pred_p = torch.norm(embedding_p - embedding_a, dim=1)
        pred_n = torch.norm(embedding_n - embedding_a, dim=1)
        pred = torch.concat([pred_p.unsqueeze(-1), pred_n.unsqueeze(-1)], dim=1)

        mean_batch_loss = F.mse_loss(pred, true)

        return mean_batch_loss


def get_SLloss(name):
    if name == "default":
        return SLLoss()
    elif name == "difficulty":
        return SLLoss_diff()
    elif name == "pos":
        return SLLoss_pos()
    elif name == "pos_difficulty":
        return SLLoss_pos_diff()
    elif name == "st2vec":
        return SLLoss_st2vec()


class SLLoss(nn.Module):
    def __init__(self):
        super(SLLoss, self).__init__()

    def forward(self, embeddings_list, ground_truth):
        pred_list = []
        for j in range(1, len(embeddings_list)):
            pred = torch.norm(embeddings_list[j] - embeddings_list[0], dim=1)
            pred_list.append(pred.unsqueeze(-1))
        pred = torch.concat(pred_list, dim=1)

        loss = F.mse_loss(pred, ground_truth)

        return loss


class SLLoss_st2vec(nn.Module):
    def __init__(self):
        super(SLLoss_st2vec, self).__init__()

    def forward(self, embeddings_list, ground_truth):
        embedding_a = embeddings_list[0]
        embedding_p = embeddings_list[1]
        embedding_n = embeddings_list[2]
        self.train_batch = len(embedding_a)

        batch_triplet_dis = ground_truth
        batch_loss = 0.0

        for i in range(self.train_batch):
            #"""
            D_ap = torch.exp(-batch_triplet_dis[i][0])
            D_an = torch.exp(-batch_triplet_dis[i][1])

            v_ap = torch.exp(-torch.dist(embedding_a[i], embedding_p[i], p=2))
            v_an = torch.exp(-torch.dist(embedding_a[i], embedding_n[i], p=2))

            loss_entire_ap = D_ap * ((D_ap - v_ap) ** 2)
            loss_entire_an = D_an * ((D_an - v_an) ** 2)

            oneloss = loss_entire_ap + loss_entire_an
            """
            D_ap = torch.tensor(batch_triplet_dis[i][0], dtype=torch.float32).cuda()
            D_an = torch.tensor(batch_triplet_dis[i][1], dtype=torch.float32).cuda()

            v_ap = torch.dist(embedding_a[i], embedding_p[i], p=2)
            v_an = torch.dist(embedding_a[i], embedding_n[i], p=2)

            oneloss = F.mse_loss(D_ap, v_ap) + F.mse_loss(D_an, v_an)
            """
            batch_loss += oneloss

        loss = batch_loss / self.train_batch

        return loss


class SLLoss_diff(nn.Module):
    def __init__(self):
        super(SLLoss_diff, self).__init__()

    def forward(self, embeddings_list, ground_truth):
        pred_list = []
        for j in range(1, len(embeddings_list)):
            pred = torch.norm(embeddings_list[j] - embeddings_list[0], dim=1)
            pred_list.append(pred.unsqueeze(-1))
        pred = torch.concat(pred_list, dim=1)


        _, true_order = torch.sort(ground_truth, dim=1)
        _, pred_order = torch.sort(pred, dim=1)

        # weight1: the learned difficulty
        weight = (pred_order - true_order).abs() + 1
        # weight2: the ground_truth_order
        #weight = 1 / (true_order + 1)
        # weight3: 1 / the ground_truth_order * the learned difficulty
        #weight = torch.mul(1 / (true_order + 1), ((pred_order - true_order).abs() + 1))

        norm = torch.norm(weight.float(), p=2, dim=1, keepdim=True)
        weight = weight / norm

        div = ground_truth - pred
        square = torch.mul(div, div)
        weight_square = torch.mul(square, weight)
        sqrt_value = torch.sqrt(torch.sum(weight_square, dim=1))

        loss = torch.mean(sqrt_value)

        #loss = F.mse_loss(pred, ground_truth)

        return loss


class SLLoss_pos(nn.Module):
    def __init__(self):
        super(SLLoss_pos, self).__init__()

    def forward(self, embeddings_list, ground_truth):
        pred_list = []
        for j in range(1, len(embeddings_list)):
            pred = torch.norm(embeddings_list[j] - embeddings_list[0], dim=1)
            pred_list.append(pred.unsqueeze(-1))
        pred = torch.concat(pred_list, dim=1)


        _, true_order = torch.sort(ground_truth, dim=1)
        _, pred_order = torch.sort(pred, dim=1)

        # weight1: the learned difficulty
        #weight = (pred_order - true_order).abs() + 1
        # weight2: the ground_truth_order
        weight = 1 / (true_order + 1)
        # weight3: 1 / the ground_truth_order * the learned difficulty
        #weight = torch.mul(1 / (true_order + 1), ((pred_order - true_order).abs() + 1))

        norm = torch.norm(weight.float(), p=2, dim=1, keepdim=True)
        weight = weight / norm

        div = ground_truth - pred
        square = torch.mul(div, div)
        weight_square = torch.mul(square, weight)
        sqrt_value = torch.sqrt(torch.sum(weight_square, dim=1))

        loss = torch.mean(sqrt_value)

        #loss = F.mse_loss(pred, ground_truth)

        return loss


class SLLoss_pos_diff(nn.Module):
    def __init__(self):
        super(SLLoss_pos_diff, self).__init__()

    def forward(self, embeddings_list, ground_truth):
        pred_list = []
        for j in range(1, len(embeddings_list)):
            pred = torch.norm(embeddings_list[j] - embeddings_list[0], dim=1)
            pred_list.append(pred.unsqueeze(-1))
        pred = torch.concat(pred_list, dim=1)


        _, true_order = torch.sort(ground_truth, dim=1)
        _, pred_order = torch.sort(pred, dim=1)

        # weight1: the learned difficulty
        #weight = (pred_order - true_order).abs() + 1
        # weight2: the ground_truth_order
        #weight = 1 / (true_order + 1)
        # weight3: 1 / the ground_truth_order * the learned difficulty
        weight = torch.mul(1 / (true_order + 1), ((pred_order - true_order).abs() + 1))

        norm = torch.norm(weight.float(), p=2, dim=1, keepdim=True)
        weight = weight / norm

        div = ground_truth - pred
        square = torch.mul(div, div)
        weight_square = torch.mul(square, weight)
        sqrt_value = torch.sqrt(torch.sum(weight_square, dim=1))

        loss = torch.mean(sqrt_value)

        #loss = F.mse_loss(pred, ground_truth)

        return loss


class SLRegressionLoss(nn.Module):
    def __init__(self):
        super(SLRegressionLoss, self).__init__()

    def forward(self, embeddings, ground_truth):
        batch_size = embeddings.shape[0]
        pred = torch.cdist(embeddings, embeddings)
        mask = torch.eye(batch_size, dtype=bool)
        mask = ~mask
        pred = pred[mask].view(batch_size, -1)
        ground_truth = ground_truth[mask].view(batch_size, -1)

        _, true_order = torch.sort(ground_truth, dim=1)
        _, pred_order = torch.sort(pred, dim=1)

        weight = (pred_order - true_order).abs() + 1
        norm = torch.norm(weight.float(), p=2, dim=1, keepdim=True)
        weight = weight / norm

        div = ground_truth - pred
        square = torch.mul(div, div)
        weight_square = torch.mul(square, weight)

        loss = torch.mean(weight_square)

        return loss
