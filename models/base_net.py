import torch
import torch.nn as nn

class SketchANet(nn.Module):

    def __init__(self, num_classes = 20):

        super(SketchANet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, (15, 15), stride=3)
        self.conv2 = nn.Conv2d(64, 128, (5, 5), stride=1)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, (7, 7), stride=1, padding=0)
        self.conv7 = nn.Conv2d(512, 512, (1, 1), stride=1, padding=0)
        self.linear = nn.Linear(512, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((3, 3), stride=2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.max_pool(x)
        x = self.dropout(self.relu(self.conv6(x)))
        x = self.dropout(self.relu(self.conv7(x)))
        x = x.view(-1, 512)
        return self.linear(x)
