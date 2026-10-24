'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    # Removed dataset_source_file_name
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        # Training Data
        X_train = []
        y_train = []
        f_train = open(self.dataset_source_folder_path + 'train.csv', 'r') # Opening train.csv
        for line in f_train:
            line = line.strip('\n') # Removing trailing newline characters
            elements = [int(i) for i in line.split(',')] # Splitting data by comma
            X_train.append(elements[:-1]) # Features are everything but the first element
            y_train.append(elements[-1]) # Label is the first element

        # Testing Data
        X_test = []
        y_test = []
        f_test = open(self.dataset_source_folder_path + 'test.csv', 'r') # Opening test.csv
        for line in f_test:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_test.append(elements[:-1]) # Features are everything but first element
            y_test.append(elements[-1]) # Label is first element

        return {'train':{'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}