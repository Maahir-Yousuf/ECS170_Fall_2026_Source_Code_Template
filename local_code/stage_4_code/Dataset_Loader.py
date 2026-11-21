'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import string
import os
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, max_len=256):
        super().__init__(dName, dDescription)
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1} # Initializing vocabulary
        self.max_len = max_len

        # ENGLISH_STOP_WORDS removes negating words, but we need to keep them
        negations = {"no", "not", "nor", "never", "neither", "none", "nothing", "nowhere", "cannot", "cant", "dont", "didnt",
                     "isnt", "wasnt", "wouldnt", "couldnt", "shouldnt"}
        self.stop_words = set(ENGLISH_STOP_WORDS) - negations# Storing the stopwords to filter them later


    # This function is for lowercase, removing punctuations, and stop words
    def clean_text(self, text):
        text = text.lower() # Lowercase
        text = re.sub(r'<.*?>', ' ', text) # Removing HTML tags
        text = text.translate(str.maketrans('', '', string.punctuation)) # Removing punctuation
        tokens = text.split()
        tokens = [w for w in tokens if w not in self.stop_words] # Filtering the stop words

        return tokens

    # Converting the .txt files to binary sentiments (pos: 1, neg: 0)
    def load_split(self, split_dir, is_train=True):
        X, y = [], []
        for label_name, label_value in [('neg', 0), ('pos', 1)]:
            folder_path = os.path.join(self.dataset_source_folder_path, split_dir, label_name)
            for filename in os.listdir(folder_path):
                if not filename.endswith(".txt"): # Ignoring files not ending with .txt
                    continue
                with open(os.path.join(folder_path, filename), 'r', encoding = 'utf-8') as f:
                    tokens = self.clean_text(f.read())

                # Converting tokens to integer indices
                seq = []

                # Looping through each cleaned word token and adding it to seq
                for word in tokens:
                    if is_train and word not in self.word_to_idx:
                        self.word_to_idx[word] = len(self.word_to_idx)
                    seq.append(self.word_to_idx.get(word, self.word_to_idx["<UNK>"]))

                if len(seq) < self.max_len:
                    seq += [self.word_to_idx["<PAD>"]] * (self.max_len - len(seq))
                else:
                    seq = seq[:self.max_len]

                X.append(seq)
                y.append(label_value)

        return X, y

    def load(self):
        print(f"Loading {self.dataset_name} data...")

        # Training and Testing Data
        X_train, y_train = self.load_split('train', is_train = True)
        X_test, y_test = self.load_split('test', is_train = False)

        # Converting to PyTorch Tensors
        X_train = torch.LongTensor(X_train)
        X_test = torch.LongTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        self.data = {'train':{'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test},'vocab': self.word_to_idx}
        return self.data


    # Building an embedding matrix
    def load_glove_embeddings(self, glove_path, embedding_dim=50):
        embedding_matrix = np.random.normal(scale=0.6,size=(len(self.word_to_idx), embedding_dim))
        embedding_matrix[0] = np.zeros((embedding_dim))

        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                word = parts[0]
                if word in self.word_to_idx:
                    try:
                        embedding_matrix[self.word_to_idx[word]] = np.array(parts[1:], dtype=np.float32)
                    except ValueError:
                        continue

        return torch.FloatTensor(embedding_matrix)