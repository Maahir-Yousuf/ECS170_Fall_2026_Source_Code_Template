from torch import nn

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import numpy as np
import matplotlib.pyplot as plt


class Method_CNN_MNIST(method, nn.Module):

    data = None
    max_epoch = 50
    learning_rate = 0.001

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Setting up Convolutional layers
        # Conv2d (in_channels, out_channels, kernel_size, stride)
        k1,k2 = 5,5
        # k1,k2 = 11,9

        self.conv1 = nn.Conv2d(1, 16, k1, 1)
        self.conv2 = nn.Conv2d(16, 16, k2, 1)

        # Activation and Pooling
        self.activation_func_1 = nn.ReLU()
        self.activation_func_2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2) # Kernel_size = 2

        # Flattening the 2D feature maps
        self.flatten = nn.Flatten()

        # Setting up Linear layers
        # Linear (in_features, out_features)

        size = ((28 - k1 +1) -k2 +1) //2
        out1, out2 = 120, 84
        # out1,out2 = 105, 30
        self.fc1 = nn.Linear(16 * size*size, out1)
        self.fc2 = nn.Linear(out1, out2)
        self.fc3 = nn.Linear(out2, 10) # Out_features must be 10 to account for all 10 digit classes


        # Forward Propagation
    def forward(self, x):
        x = self.activation_func_1(self.conv1(x))
        x = self.activation_func_2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.activation_func_1(self.fc1(x))
        x = self.activation_func_2(self.fc2(x))
        x = self.fc3(x)
        return x

    # Instead of using one single batch and crashing our computers, we will implement mini-batching

    # Training (Copied from Method_MLP)
    def train(self, X, y):

        # Setting up MPS (Deep learning for Mac Apple Silicon)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        X = torch.as_tensor(X, dtype=torch.float32).to(device)
        y = torch.as_tensor(y, dtype=torch.long).to(device)

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # For applying learned model to testing set, we need to include keep track of the loss values
        # A list for the lost values
        loss_history = []

        batch_size = 128
        num_samples = X.shape[0]

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            epoch_loss = 0.0

            permutation = torch.randperm(num_samples)

            # Mini-batch loop
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i + batch_size]
                X_batch = X[indices]
                y_batch = y[indices]

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            # Average loss for epoch
            avg_loss = epoch_loss / num_samples

            # Recording loss for each epoch
            loss_history.append(avg_loss)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function

            if (epoch + 1) % 5 == 0 or epoch == 0:
                accuracy_evaluator.data = {'true_y': y_batch.cpu(), 'pred_y': y_pred.cpu().max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Avg_epoch_Loss:', avg_loss)

        # The Convergence Curve Plot
        plt.figure(figsize=(8, 6))
        plt.plot(range(self.max_epoch), loss_history, color='red')  # X axis are the epochs, Y axis are the loss values
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def test(self, X):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        X = torch.as_tensor(X, dtype=torch.float32).to(device)

        batch_size = 128
        num_samples = X.shape[0]
        all_preds = []

        # Saving memory by disabling gradient calculation
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_X = X[i:i + batch_size]
                y_pred = self.forward(batch_X)
                all_preds.append(y_pred.cpu().max(1)[1])

        # Putting all batch predictions into one array
        return torch.cat(all_preds)



    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}











