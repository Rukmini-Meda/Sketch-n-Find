import time
import datetime
import pytz
import os
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from models.triplet_net import TripletNet
from data.data_loaders import TestDataLoader
from data_preparation import TestDataPreparator
from utils import *

def test(batch_size, sketch_dataloader, photo_dataloader, model, k = 5, num_display = 2):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # images_model = images_model.to(device)
    # sketches_model = sketches_model.to(device)
    # images_model.eval()
    # sketches_model.eval()
    model = model.to(device)
    model.eval()

    print("Processing the images. Batch size: %d; Number of batches: %d" %(batch_size, len(photo_dataloader)))
    start_time = time.time()
    image_feature_predictions = []
    image_labels = []
    test_images = []

    with torch.no_grad():
        for iteration, batch in enumerate(photo_dataloader):
            images, labels = batch
            images = torch.autograd.Variable(images.to(device))
            pred_features = model(images)
            test_images.append(images)
            image_feature_predictions.append(pred_features)
            image_labels.append(labels)
    image_feature_predictions = torch.cat(image_feature_predictions, dim = 0)
    image_labels = torch.cat(image_labels, dim = 0)
    test_images = torch.cat(test_images, dim = 0)

    end_time = time.time()

    print("Processed the images. Time taken: %s" %(str(datetime.timedelta(seconds = int(end_time - start_time)))))

    print("Processing the sketches. Batch size: %d, Number of batches: %d" %(batch_size, len(sketch_dataloader)))
    start_time = time.time()
    sketch_feature_predictions = []
    sketch_labels = []
    test_sketches = []

    with torch.no_grad():
        for iteration, batch in enumerate(sketch_dataloader):
            sketches, labels = batch
            sketches = torch.autograd.Variable(sketches.to(device))
            pred_features = model(sketches)
            test_sketches.append(sketches)
            sketch_feature_predictions.append(pred_features)
            sketch_labels.append(labels)

    sketch_feature_predictions = torch.cat(sketch_feature_predictions, dim = 0)
    sketch_labels = torch.cat(sketch_labels, dim = 0)
    test_sketches = torch.cat(test_sketches, dim = 0)
    end_time = time.time()

    print("Processed the sketches. Time taken: %s" %(str(datetime.timedelta(seconds = int(end_time - start_time)))))

    image_feature_predictions = image_feature_predictions.cpu().numpy()
    sketch_feature_predictions = sketch_feature_predictions.cpu().numpy()
    image_labels = image_labels.cpu().numpy()
    sketch_labels = sketch_labels.cpu().numpy()

    distance = cdist(sketch_feature_predictions, image_feature_predictions, 'minkowski')
    similarity = 1.0/distance

    is_correct_label = 1 * (np.expand_dims(sketch_labels, axis = 1) == np.expand_dims(image_labels, axis = 0))

    # print(sketch_labels.shape)
    # print(is_correct_label)
    # print(similarity)
    average_precision_scores = []
    for i in range(sketch_labels.shape[0]):
        # print(is_correct_label[i])
        # print(similarity[i])
        temp = average_precision_score(is_correct_label[i], similarity[i])
        if temp != None and temp == temp:
            average_precision_scores.append(temp)
        else:
            # tp = is_correct_label[i].sum()
            # fp = len(is_correct_label[i]) - tp
            print(is_correct_label[i])
            print(similarity[i])
            print(temp)
            # return
            average_precision_scores.append(0)
    average_precision_scores = np.array(average_precision_scores)

    # index2label = {v: k for k, v in label2index.items()}
    for classes in set(sketch_labels):
        print('Class: %s, mAP: %f' %(classes, average_precision_scores[sketch_labels == classes].mean()))
    mean_average_precision = average_precision_scores.mean()
    sketches, image_grids = get_sketch_images_grids(test_sketches, test_images, similarity, k, num_display)
    return sketches, image_grids, mean_average_precision

if __name__ == '__main__':
    print("Testing the model ...")
    # num_images = int(input("Number of random images to output for every sketch"))
    # num_sketches = int(input("Number of random sketches to output"))
    batch_size = int(input("Batch size to process the test sketches/photos"))
    # model_path = input("Enter the path to load trained model")
    # output_dir = input("Enter the path to the output folder")

    args = {
        "num_images": 5,
        "num_sketches": 2,
        "batch_size": batch_size,
        # "output_dir": output_dir,
        "model": "trained_models/CP_B40_E8_LD_DQ_TV60/last.pth.tar"
    }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data = TestDataPreparator()
    test_sketch_data, test_photo_data = data.get_dataset()
    dataloader = TestDataLoader(test_sketch_data, test_photo_data, args["batch_size"])
    sketch_test_dataloader, photo_test_dataloader = dataloader.get_loader()

    # image_model = SketchANet().to(device)
    # sketch_model = SketchANet().to(device)
    model = TripletNet().to(device)

    if args["model"]:
        load_checkpoint(args["model"], model)

    test_dict = {}
    sketches, image_grids, test_mAP = test(args["batch_size"], sketch_test_dataloader, photo_test_dataloader, model)
    print(f"Average test mAP: {test_mAP}")

    # if not os.path.isdir(args["output_dir"]):
    #     os.mkdir(args["output_dir"])

    # for i in range(args["num_sketches"]):
    #     # sketches[i] = np.squeeze(sketches[i], axis=2)  # axis=2 is channel dimension 
    #     # image_grids[i] = np.squeeze(image_grids[i], axis = 2)
    #     Image.fromarray(np.uint8(sketches[i] * 255)).save(os.path.join(args["output_dir"], "Sketch_%d.png"%(i)))
    #     Image.fromarray(np.uint8(image_grids[i] * 255)).save(os.path.join(args["output_dir"], "Images_%d.png"%(i)))