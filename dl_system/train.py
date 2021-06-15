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
from data.data_loaders import TrainDataLoader, ValDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# from prepare_data import train_data, val_data
from data_preparation import TrainDataPreparator, ValDataPreparator

class ModelTrainer():

    def __init__(self):
        self.dataloader = None

    def train_validate(self, train_dataloader, val_dataloader, config, checkpoint = None):
        # print("Entered the train function")
        num_batches = len(train_dataloader)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = TripletNet().to(device)
        # print("Model is created")
        params = [param for param in model.parameters() if param.requires_grad == True]
        # print("params obtained as:")
        # print(params)
        optimizer = torch.optim.Adam(params, lr=config["lr"])
        # print("Optimizer created")
        criterion = nn.TripletMarginLoss(margin=1.0, p = 2)
        # print("Loss function created")
        if checkpoint:
            load_checkpoint(checkpoint, model, optimizer)

        print("Training ...")

        losses = []
        val_losses = []
        losses_t = []
        val_losses_t = []

        total_iterations = 0
        total_val_iterations = 0

        for epoch in range(config['epochs']):
            print(f"Epoch #{epoch}")
            accumulated_triplet_loss = RunningAverage()
            accumulated_iteration_time = RunningAverage()
            print(f"Loss accumulated = {accumulated_triplet_loss()}")
            print(f"Time so far = {accumulated_iteration_time()}")
            epoch_start_time = time.time()

            model.train()
            triplet_loss = 0
            # Training
            for iteration, batch in tqdm(enumerate(train_dataloader)):
                print(f"Iteration #{iteration}")
                time_start = time.time()

                anchors, positives, negatives = batch
                anchors = torch.autograd.Variable(anchors.to(device))
                positives = torch.autograd.Variable(positives.to(device))
                negatives = torch.autograd.Variable(negatives.to(device))
                # print("Torch variables created")
                predicted_sketch_features, predicted_positives_features, predicted_negatives_features = model(anchors, positives, negatives)
                # print("Model instantiated")
                triplet_loss = criterion(predicted_sketch_features, predicted_positives_features, predicted_negatives_features)
                accumulated_triplet_loss.update(triplet_loss, anchors.shape[0])
                
                # print("Loss updated")
                optimizer.zero_grad()
                # print("Made grad of optimizer zero")
                triplet_loss.backward()
                # print("Doing backward prop")
                optimizer.step()
                # print("Doing optimizer steps")
                time_end = time.time()
                accumulated_iteration_time.update(time_end - time_start)
                # print("Accumulated iteration time")
                if iteration % config['print_every'] == 0:
                    eta_cur_epoch = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))
                    print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')).replace(microsecond = 0), end = ' ')
                    print("Epoch: %d [%d / %d] ; eta: %s" %(epoch, iteration, num_batches, eta_cur_epoch))
                    print('Average Triplet Loss: %f(%f);' %(triplet_loss, accumulated_triplet_loss()))
                total_iterations += 1
            print('Average Triplet Loss: %f(%f);' %(triplet_loss, accumulated_triplet_loss()))
            losses_t.append(triplet_loss.detach().numpy())
            losses.append(accumulated_triplet_loss().detach().numpy()) 
            if not os.path.exists(config['checkpoint_dir']):
                os.mkdir(config["checkpoint_dir"])
            save_checkpoint({
                'iteration': iteration + epoch * num_batches,
                'model': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, checkpoint_dir = config['checkpoint_dir'])
            print('Saved epoch!')
            print('\n\n\n')

            print("Validating on the validation dataset")
            # Validation
            model.eval()

            with torch.no_grad():
                eval_losses = RunningAverage()
                eval_time = RunningAverage()
                triplet_loss = 0
                for iteration, batch in tqdm(enumerate(val_dataloader)):
                    print(f"Iteration #{iteration}")
                    time_start = time.time()

                    anchors, positives, negatives = batch
                    anchors = torch.autograd.Variable(anchors.to(device))
                    positives = torch.autograd.Variable(positives.to(device))
                    negatives = torch.autograd.Variable(negatives.to(device))
                    # print("Torch variables created")
                    predicted_sketch_features, predicted_positives_features, predicted_negatives_features = model(anchors, positives, negatives)
                    # print("Model instantiated")
                    triplet_loss = criterion(predicted_sketch_features, predicted_positives_features, predicted_negatives_features)
                    eval_losses.update(triplet_loss, anchors.shape[0])
                    

                    time_end = time.time()
                    eval_time.update(time_end - time_start)
                    total_val_iterations += 1

                print(f"Average validation triplet loss is {triplet_loss}({eval_losses()})")
                val_losses.append(eval_losses().detach().numpy())
                val_losses_t.append(triplet_loss.detach().numpy())

            epoch_end_time = time.time()
            print("Epoch %d complete, time taken: %s" %(epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
            torch.cuda.empty_cache()
            print("Validation done!")
            

        plt.plot(list(range(epochs)), losses, label= "Average training loss")
        plt.plot(list(range(epochs)), val_losses, label = "Average validation loss")
        plt.plot(list(range(epochs)), losses_t, label = "Training loss")
        plt.plot(list(range(epochs)), val_losses_t, label = "Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.savefig("plots/plot2.png")
        plt.show()

if __name__ == '__main__':
    print("Training the model ...")
    batch_size = int(input("Enter batch size for training:"))
    epochs = int(input("Enter number of epochs for training:"))
    path = input("Enter the directory name to store the checkpoints of the model")
    # lr = int(input("Enter the learning rate"))

    args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "print_every": 10,
        "checkpoint_dir": path,
        "lr": 0.001
    }
    

    data = TrainDataPreparator()
    train_sketch_data, train_photo_data = data.get_dataset()
    data = ValDataPreparator()
    val_sketch_data, val_photo_data = data.get_dataset()

    batch_size = args["batch_size"]

    dataloader = TrainDataLoader(train_sketch_data, train_photo_data, batch_size)
    train_dataloader = dataloader.get_loader()
    
    dataloader = ValDataLoader(val_sketch_data, val_photo_data, batch_size)
    val_dataloader = dataloader.get_loader()

    trainer = ModelTrainer()
    trainer.train_validate(train_dataloader, val_dataloader, args)