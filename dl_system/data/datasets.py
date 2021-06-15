from torch.utils.data import Dataset
from data_preparation import create_utils
import numpy as np

class SketchyTripletDataset(Dataset):

    def __init__(self, sketch_dataset, photo_dataset):
        self.sketches_dataset = sketch_dataset
        self.photos_dataset = photo_dataset
        self.sketches_classes, self.sketches_class_counts, self.sketches_class_images_list = create_utils(self.sketches_dataset)
        self.photos_classes, self.photos_class_counts, self.photos_class_images_list = create_utils(self.photos_dataset)
        
    def __len__(self):
        return len(self.sketches_dataset)

    def __getitem__(self, index):
        sketch = self.sketches_dataset[index]
        class_label = sketch[1]
        sketch_image = sketch[0]
        # num_classes = self.sketches_classes
        # print(f"Class label is {class_label}")
        # print(f"Number of classes is {num_classes}")
        labels_list = list(self.photos_class_counts.keys())
        # print(len(labels_list))
        while True:
            label = np.random.choice(labels_list)
            if label != class_label:
                break
        # print(f"Negative label is {label}")
        # print(f"Class label is {class_label}")
        # print(f"Negative label is {label}")
        # print(len(self.photos_class_images_list[label]))
        positive = self.get_image_for_class(class_label)
        negative = self.get_image_for_class(label)
        # print("Returning a dataset item")
        return sketch_image, positive, negative

    def get_image_for_class(self,label):
        n = self.photos_class_counts[label]
        # print(n)
        index = np.random.choice(list(range(n)))
        # print(index)
        return self.photos_class_images_list[label][index]