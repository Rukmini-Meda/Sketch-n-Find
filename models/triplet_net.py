import torch
import torch.nn as nn
from models.base_net import SketchANet

class TripletNet(nn.Module):

    def __init__(self, num_classes = 125):
        super(TripletNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_sketchanet = SketchANet(self.num_classes)

    def forward(self, anchor, positive = None, negative = None):
        x_a = self.embedding_sketchanet(anchor)
        if (positive != None) and (negative != None):
            x_p = self.embedding_sketchanet(positive)
            x_n = self.embedding_sketchanet(negative)
            return x_a, x_p, x_n
        else:
            return x_a

    