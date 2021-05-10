from torch.utils.data import DataLoader
from data.datasets import SketchyTripletDataset

class TrainDataLoader():

    def __init__(self, sketch_dataset, photo_dataset, batch_size):
        self.train_dataset = SketchyTripletDataset(sketch_dataset, photo_dataset)
        self.loader = DataLoader(dataset = self.train_dataset, shuffle = True, batch_size = batch_size, drop_last=True)

    def get_loader(self):
        # print("Returned the train loader")
        return self.loader

class ValDataLoader():

    def __init__(self, sketch_dataset, photo_dataset, batch_size):
        self.val_dataset = SketchyTripletDataset(sketch_dataset, photo_dataset)
        self.loader = DataLoader(dataset = self.val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def get_loader(self):
        return self.loader

class TestDataLoader():

    def __init__(self, sketch_dataset, photo_dataset, batch_size):
        self.sketch_loader = DataLoader(dataset = sketch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.photo_loader = DataLoader(dataset = photo_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def get_loader(self):
        return self.sketch_loader, self.photo_loader
