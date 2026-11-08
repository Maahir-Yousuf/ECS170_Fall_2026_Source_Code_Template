'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import pickle
import numpy as np
import torch
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None, image_type="MNIST"):
        super().__init__(dName, dDescription)
        self.image_type = image_type
    
    def load(self):
        print(f"Loading {self.dataset_name} data...")
        filepath = self.dataset_source_folder_path + self.dataset_source_file_name

        with open(filepath, 'rb') as f: # Opening file in read binary mode
            raw_data = pickle.load(f) # Reconstructing object

        # Training Data
        X_train = []
        y_train = []

        # Processing Training Data
        for instance in raw_data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])

        # Testing Data
        X_test = []
        y_test = []

        # Processing Testing Data
        for instance in raw_data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])

        # Creating NumPy Arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Changing image tensors to follow the (Batch, Channels, Height, Width) format
        if self.image_type == "MNIST": # (Batch, 1, 28, 28)
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        elif self.image_type == "CIFAR": # (Batch, 3, 32, 32)
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.transpose(X_test, (0, 3, 1, 2))

        elif self.image_type == "ORL": # (Batch, 1, 112, 92) using only the R channel
            X_train = X_train[:, :, :, 0] # Extracting R channel
            X_test = X_test[:, :, :, 0]
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        # Converting to PyTorch Tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)

        self.data = {'train':{'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

        return self.data