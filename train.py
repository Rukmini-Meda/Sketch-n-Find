import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TripletNet()
triplet_loss = nn.TripletMarginLoss(margin=0.1)
optimizer = optim.Adam(model.parameters())
epoch = 1

for index, (anchor, positive, negative) in enumerate(listed):
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)
    anchor_features, positive_features, negative_features = model(anchor, positive, negative)
    loss = triplet_loss(anchor, positive, negative)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Index #{index}')
    print(f'Epoch #{epoch}')
    # print(f'Loss: {loss}')
    epoch += 1


