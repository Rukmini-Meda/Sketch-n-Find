import numpy as np
import torch
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import random_split, DataLoader, Dataset
from utils import download_extract_sketchy_dataset

class TrainDataLoader():

    def __init__(self, batch_size):
        self.train_dataset = SketchyTripletDataset()
        self.loader = DataLoader(dataset = self.train_dataset, shuffle = True, batch_size = batch_size)

    def get_loader(self):
        return self.loader

class SketchyTripletDataset(Dataset):

    def __init__(self):
        data = DataPreparator()
        self.sketches_dataset, self.photos_dataset = data.get_train_datasets()
        self.sketches_classes, self.sketches_class_counts, self.sketches_class_images_list = create_utils(self.sketches_dataset)
        self.photos_classes, self.photos_class_counts, self.photos_class_images_list = create_utils(self.photos_dataset)
        
    def __len__(self):
        return len(self.sketches_dataset) + len(self.photos_dataset)

    def __getitem__(self, index):
        sketch = self.sketches_dataset[index]
        class_label = sketch[1]
        sketch_image = sketch[0]
        num_classes = self.sketches_classes
        print(f"Class label is {class_label}")
        print(f"Number of classes is {num_classes}")
        while True:
            label = np.random.choice(list(range(1, num_classes+1)))
            if label != class_label:
                break
        print(f"Negative label is {label}")
        positive = self.get_image_for_class(class_label, self.photos_class_images_list)
        negative = self.get_image_for_class(label, self.photos_class_images_list)
        return sketch_image, positive, negative

    def get_image_for_class(self,label, class_images_list):
        n = len(class_images_list[label])
        index = np.random.choice(list(range(n)))
        return class_images_list[label][index]

class DataPreparator():
    
    def __init__(self):
        # self.output_folder = "dataset/rendered_256x256"
        self.output_folder = "dataset"
        self.id = "tx_000000000000"
        if not self._check_exists():
            download_extract_sketchy_dataset()
        self.sketch_dir = self.output_folder + "/" + "256x256/photo/" + self.id
        self.photo_dir = self.output_folder + "/" + "256x256/sketch/" + self.id
        self.prepare_data()

    def _check_exists(self):
        return os.path.exists(self.output_folder)
        
    def get_train_datasets(self):
        return self.sketches_train_dataset, self.photos_train_dataset

    def get_test_datasets(self):
        return self.sketches_test_dataset, self.photos_test_dataset

    def prepare_data(self):
        np.random.seed(0)
        torch.manual_seed(0)

        print(f"Photos are here -> {self.photo_dir}")
        print(f"Sketches are here -> {self.sketch_dir}")

        image_transforms = {
            "train": transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
        }

        sketches_dataset = datasets.ImageFolder(root = self.sketch_dir, transform=image_transforms["train"])
        photos_dataset = datasets.ImageFolder(root = self.photo_dir, transform=image_transforms["train"])

        # print(sketches_dataset.class_to_idx)
        # print(photos_dataset.class_to_idx)
        
        # print(f"Distribution of classes in sketch dataset: \n{self.sketches_class_counts}")
        # print(f"Distribution of classes in photo dataset: \n{self.photos_class_counts}")

        total = len(sketches_dataset)
        split_1 = total//10
        split_2 = total - split_1
        print(split_1)
        print(split_2)
        print(total)

        self.sketches_train_dataset, self.sketches_test_dataset = random_split(sketches_dataset, (split_2, split_1))

        total = len(photos_dataset)
        split_1 = total//10
        split_2 = total - split_1
        print(split_1)
        print(split_2)
        print(total)
        self.photos_train_dataset, self.photos_test_dataset = random_split(photos_dataset, (split_2, split_1))

        print(len(self.photos_train_dataset))
        print(len(self.sketches_train_dataset))
        

def create_utils(dataset):
    print("Creating utils")
    # I need class - count map
    # I need class - images list map
    class_to_idx = {}
    index = 0
    for _, l in dataset:
        if l in class_to_idx:
           continue
        class_to_idx[l] = index 
        index += 1
    class_counts = {k: 0 for k, _ in class_to_idx.items()}
    class_images_list = {k: [] for k, _ in class_to_idx.items()}
    total_classes = len(class_to_idx.items())

    for image, label in dataset:
        class_counts[label] += 1
        class_images_list[label].append(image)
    print("Done!")
    return total_classes, class_counts, class_images_list