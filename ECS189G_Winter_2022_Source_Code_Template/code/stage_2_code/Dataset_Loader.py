from code.base_class.dataset import dataset
import pandas as pd
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        """
        Loads the training and testing data into a dictionary.

        :returns: Dictionary of both datasets.
        """
        print('Loading data...!!!!')

        f = open(self.dataset_source_folder_path + "train.csv", 'r')
        data = np.loadtxt(f, delimiter=",")
        X_train = data[:, 1:]
        y_train = data[:, 0]
        f.close()

        f = open(self.dataset_source_folder_path + "test.csv", 'r')
        data = np.loadtxt(f, delimiter=",")
        X_test = data[:, 1:]
        y_test = data[:, 0]
        f.close()

        return {'X_train': X_train, 'y_train': y_train, 'X_test' : X_test, 'y_test' : y_test}