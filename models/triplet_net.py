import torch
import torch.nn as nn
from base_net import SketchANet

class TripletNet(nn.Module):

    def __init__(self, num_classes = 20):
        super(TripletNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_net = SketchANet(self.num_classes)

    def forward(self, anchor, positive, negative):
        x_a = self.embedding_net(anchor)
        x_p = self.embedding_net(positive)
        x_n = self.embedding_net(negative)
        return x_a, x_p, x_n
