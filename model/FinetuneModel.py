import torch.nn as nn


class SimilarityLearning(nn.Module):
    def __init__(self, pretrained_model, input_size):
        super(SimilarityLearning, self).__init__()
        self.pretrained_model = pretrained_model
        self.encoder = nn.Sequential(nn.Linear(input_size, input_size),
                                     nn.ReLU(),
                                     nn.Linear(input_size, input_size))

    def forward(self, s, t, rs, lengths, st):
        h, z = self.pretrained_model(s, t, rs, lengths, st)
        return self.encoder(h)