from code.base_class.dataset import dataset
import pickle
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)

        training_X = []
        training_y = []
        for image_lab in data["train"]:
            image = image_lab["image"]
            label = image_lab["label"]
            training_X.append(image)
            training_y.append(label)

        testing_X = []
        testing_y = []
        for image_lab in data["test"]:
            image = image_lab["image"]
            label = image_lab["label"]
            testing_X.append(image)
            testing_y.append(label)

        # Normalize the values
        training_X = torch.Tensor(np.array(training_X))
        testing_X = torch.Tensor(np.array(testing_X))

        # This normalization is important for proper rendering on matplotlib for float-valued arrays
        training_X /= 255
        testing_X /= 255

        return {"X_train": training_X, "y_train": np.array(training_y),
                "X_test": testing_X, "y_test": np.array(testing_y)}