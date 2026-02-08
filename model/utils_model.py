import torch
from torch import nn
import copy
import numpy as np
import pickle
import math
from pynvml import *
import psutil


def clones(module, N):
    """
    Produce N identical layers.
    :param module:
    :param N:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def mask_padding(lengths, max_len=None):
    """
    return padding mask based on lengths
    Note that only allow padding in the tail
    :param lengths: (batch_size,)
    :return: mask, bool tensor, (batch_size, max_num_seq), padding position is True
    """
    if max_len is None:
        max_len = int(lengths.max())
    mask = torch.ones((lengths.size()[0], max_len)).cuda()
    for i, len in enumerate(lengths):
        mask[i, len:] = 0
    return mask == 0


def get_parameter_number(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def load_pretrained(filepath, vocab_size):
    if filepath.endswith("txt"):
        with open(filepath) as f:
            # jump the first line, which is the description of the embeddings.
            n_words, dim = [int(value) for value in f.readline().strip().split(" ")]
            embeddings = np.random.randn(vocab_size, dim)
            for line in f:
                word_vec = [float(value) for value in line.strip().split(" ")]
                embeddings[int(word_vec[0])] = word_vec[1:]
        embeddings = torch.tensor(embeddings, dtype=torch.float)
    elif filepath.endswith("pkl"):
        with open(filepath, "rb") as f:
            embeddings = pickle.load(f)

    return embeddings


def init_parameters(model):
    from torch.nn import init
    for param in model.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)


def seq_pad(seq_list, max_length=None, padding_idx=0):
    if max_length is None:
        seq_lengths = [len(seq) for seq in seq_list]
        max_length = max(seq_lengths)
    for seq in seq_list:
        seq += [padding_idx] * (max_length - len(seq))

    return seq_list


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 2)
        self.w_2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


def adjust_learning_rate(optimizer, lr, epoch, nepoch):
    # Decay the learning rate based on cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / nepoch)) # in range (0, 1] * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


nvmlInit()
class GPUInfo:

    _h = nvmlDeviceGetHandleByIndex(0)

    @classmethod
    def mem(cls):
        info = nvmlDeviceGetMemoryInfo(cls._h)
        return info.used // 1048576, info.total // 1048576 # in MB

class RAMInfo:
    @classmethod
    def mem(cls):
        return int(psutil.Process(os.getpid()).memory_info().rss / 1048576) # in MB
