# RNN Model copied from CNN stage 3

from torch import nn

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import numpy as np
import matplotlib.pyplot as plt


class Method_RNN(method, nn.Module):

    data = None
    max_epoch = 10
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, vocab_size, embedding_dim=50, hidden_size=128,
                 output_dim=2, num_layers=2, dropout=0.3, rnn_type='RNN', pretrained_embeddings=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Converting everything to uppercase
        self.rnn_type = rnn_type.upper()

        # Embedding Layer (GloVe)
        if pretrained_embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Recurrent Layer
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)

        # Dropout and Linear layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)


    # Forward Propagation
    # For this, we need to make sure the memory is not completely overwritten with zeros
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()

        embedded = self.dropout(self.embeddings(x))
        out, _ = self.rnn(embedded)

        masked_out = out * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = masked_out.sum(dim=1) / lengths

        return self.fc(self.dropout(pooled))

    # Training
    def train(self, X, y):

        # Setting up MPS (Deep learning for Mac Apple Silicon)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        nn.Module.train(self, True)
        X = torch.as_tensor(X, dtype=torch.long).to(device)
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

        batch_size = 64
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
                nn.utils.clip_grad_norm_(self.parameters(),max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            # Average loss for epoch
            avg_loss = epoch_loss / num_samples

            # Recording loss for each epoch
            loss_history.append(avg_loss)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function

            if (epoch + 1) % 2 == 0 or epoch == 0:
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
        nn.Module.train(self, False)
        X = torch.as_tensor(X, dtype=torch.long).to(device)

        batch_size = 64
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











