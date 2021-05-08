import time
import datetime
import pytz
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
# from evaluate import evaluate
from utils import *
from models.triplet_net import TripletNet
from data_preparation import TrainDataLoader
from tqdm import tqdm

class ModelTrainer():

    def __init__(self):
        self.dataloader = None

    def train(self, config, checkpoint = None):
        batch_size = config["batch_size"]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataloader = TrainDataLoader(batch_size)
        train_dataloader = self.dataloader.get_loader()
        num_batches = len(train_dataloader)

        model = TripletNet().to(device)

        params = [param for param in model.parameters() if param.requires_grad == True]
        
        optimizer = torch.optim.Adam(params, lr=config["lr"])
        criterion = nn.TripletMarginLoss(margin=1.0, p = 2)

        if checkpoint:
            load_checkpoint(checkpoint, model, optimizer)

        print("Training ...")

        for epoch in range(config['epochs']):
            print(f"Epoch #{epoch}")
            accumulated_triplet_loss = RunningAverage()
            accumulated_iteration_time = RunningAverage()
            print(f"Loss accumulated = {accumulated_triplet_loss}")
            print(f"Time so far = {accumulated_iteration_time}")
            epoch_start_time = time.time()

            model.train()

            for iteration, batch in tqdm(enumerate(train_dataloader)):
                print(f"Iteration #{iteration}")
                time_start = time.time()

                anchors, positives, negatives = batch
                anchors = torch.autograd.Variable(anchors.to(device))
                positives = torch.autograd.Variable(positives.to(device))
                negatives = torch.autograd.Variable(negatives.to(device))

                predicted_sketch_features, predicted_positives_features, predicted_negatives_features = model(anchors, positives, negatives)

                triplet_loss = criterion(predicted_sketch_features, predicted_positives_features, predicted_negatives_features)
                accumulated_triplet_loss.update(triplet_loss, anchors.shape[0])

                optimizer.zero_grad()
                triplet_loss.backward()
                optimizer.step()

                time_end = time.time()
                accumulated_iteration_time.update(time_end - time_start)

                if iteration % config['print_every'] == 0:
                    eta_cur_epoch = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))
                    print(datetime.datetime.now(pytx.timezone('Asia/Kolkata')).replace(microsecond = 0), end = ' ')
                    print("Epoch: %d [%d / %d] ; eta: %s" %(epoch, iteration, num_batches, eta_cur_epoch))
                    print('Average Triplet Loss: %f(%f);' %(triplet_loss, accumulated_triplet_loss()))
                
            epoch_end_time = time.time()
            print("Epoch %d complete, time taken: %s" %(epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
            torch.cuda.empty_cache()

            save_checkpoint({
                'iteration': iteration + epoch * num_batches,
                'model': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, checkpoint_dir = config['checkpoint_dir'])
            print('Saved epoch!')
            print('\n\n\n')

if __name__ == '__main__':
    print("Training the model ...")
    batch_size = int(input("Enter batch size for training:"))
    epochs = int(input("Enter number of epochs for training:"))

    args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "print_every": 10,
        "checkpoint_dir": "pretrained models",
        "lr": 0.01
    }

    trainer = ModelTrainer()
    trainer.train(args)