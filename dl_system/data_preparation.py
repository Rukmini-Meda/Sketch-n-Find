import numpy as np
import torch
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import random_split, DataLoader, Dataset
from utils import download_extract_sketchy_dataset

class TrainDataPreparator():

    def __init__(self):
        self.sketch_folder = "tenth_sketchy_data/train/sketch"
        self.photo_folder = "tenth_sketchy_data/train/photo"
        self.prepare_data()
    
    def get_dataset(self):
        return self.sketches_dataset, self.photos_dataset

    def prepare_data(self):
        print(f"Photos are here -> {self.photo_folder}")
        print(f"Sketches are here -> {self.sketch_folder}")

        transform =  transforms.Compose([
                transforms.Resize((225,225)),
                transforms.ToTensor()
            ])

        self.sketches_dataset = datasets.ImageFolder(root = self.sketch_folder, transform=transform)
        self.photos_dataset = datasets.ImageFolder(root = self.photo_folder, transform=transform)

class ValDataPreparator():

    def __init__(self):
        self.sketch_folder = "tenth_sketchy_data/val/sketch"
        self.photo_folder = "tenth_sketchy_data/val/photo"
        self.prepare_data()
    
    def get_dataset(self):
        return self.sketches_dataset, self.photos_dataset

    def prepare_data(self):
        print(f"Photos are here -> {self.photo_folder}")
        print(f"Sketches are here -> {self.sketch_folder}")

        transform =  transforms.Compose([
                transforms.Resize((225,225)),
                transforms.ToTensor()
            ])

        self.sketches_dataset = datasets.ImageFolder(root = self.sketch_folder, transform=transform)
        self.photos_dataset = datasets.ImageFolder(root = self.photo_folder, transform=transform)

class TestDataPreparator():

    def __init__(self):
        self.sketch_folder = "quarter_sketchy_data/test/sketch"
        self.photo_folder = "quarter_sketchy_data/test/photo"
        self.prepare_data()
    
    def get_dataset(self):
        return self.sketches_dataset, self.photos_dataset

    def prepare_data(self):
        print(f"Photos are here -> {self.photo_folder}")
        print(f"Sketches are here -> {self.sketch_folder}")

        transform =  transforms.Compose([
                transforms.Resize((225,225)),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

        self.sketches_dataset = datasets.ImageFolder(root = self.sketch_folder, transform=transform)
        self.photos_dataset = datasets.ImageFolder(root = self.photo_folder, transform=transform)

def create_utils(dataset):
    print("----")
    print("Creating utils")
    
    # I need class - count map
    # I need class - images list map
    
    class_counts = {}
    class_images_list = {}
    total_classes = 0
    # print(class_to_idx)
    for image, label in dataset:
        # print(label)
        if label in class_counts:
            class_counts[label] += 1
            class_images_list[label].append(image)
        else:
            class_counts[label] = 1
            class_images_list[label] = []
            class_images_list[label].append(image)
            total_classes += 1
    print("Done!")
    print("----")
    return total_classes, class_counts, class_images_list