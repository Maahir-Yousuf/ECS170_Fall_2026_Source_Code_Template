
from torch import nn
import torch
import matplotlib.pyplot as plt
from local_code.base_class.method import method

from local_code.stage_4_code.Dataset_Loader_Generator import  pad_id, unk_id, eos_id, detokenize

class Method_RNN_Generator(method, nn.Module):

    data = None
    epochs = 80
    learning_rate = 1e-13

    start_phrases = [
        ['what', 'did', 'the'],
        ['why', 'did', 'the'],
        ['how', 'did', 'the'],
        ['how', 'do', 'you'],
        ['did', 'you', 'know'],
        ['that', 'car', 'is'],
    ]

    def __init__(self, mName, mDescription, vocab_size, embedding_dim=100, hidden_size=256,
                 num_layers=2, dropout=0.3, rnn_type='LSTM', pretrained_embeddings=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type.upper()

        if pretrained_embeddings is not None:
            embedding_dim = pretrained_embeddings.shape[1]
            self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze = False, padding_idx = pad_id)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx= pad_id)


        rnn_dropout = dropout if num_layers > 1 else 0.0
        cell = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[self.rnn_type]
        self.rnn = cell(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = rnn_dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)



    def forward(self, x, hidden= None):
        emb = self.dropout(self.embeddings(x))
        output, hidden = self.rnn(emb, hidden)
        logits = self.fc(self.dropout(output))
        return logits, hidden



    def train(self, x, y):

        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available()
                              else 'cpu')

        self.to(device)
        nn.Module.train(self, True)
        x = torch.as_tensor(x, dtype=torch.long).to(device)
        y = torch.as_tensor(y, dtype=torch.long).to(device)


        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        loss_func = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss_history = []
        batch_size, n = 64, x.shape[0]


        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            total_token = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits,_ = self.forward(x[idx])

                loss = loss_func(logits.reshape(-1, logits.size(-1)), y[idx].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                n_token = (y[idx] != pad_id).sum().item()
                epoch_loss += loss.item() * n_token
                total_token += n_token

            avg = epoch_loss /total_token
            loss_history.append(avg)
            print("Epoch:", epoch, "Loss:", avg, "Perplexity:", torch.exp(torch.tensor(avg)).item())

        plt.plot(range(self.epochs), loss_history, color="red")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


    @torch.no_grad()

    def generate(self, start_ids, word2idx, idx2word, max_len=100, temperature=0.5):

        device = next(self.parameters()).device
        nn.Module.train(self, False)

        ids = [word2idx.get(w.lower(), unk_id) for w in start_ids]
        out = list(start_ids)

        logits, hidden = self.forward(torch.tensor([ids], dtype=torch.long, device=device))


        for _ in range(max_len):

            l = logits[0,-1]/ temperature

            l[pad_id] = float('-inf')
            l[unk_id] = float('-inf')
            nxt = torch.multinomial(torch.softmax(l, dim=-1),1).item()
            if nxt == eos_id:
                break

            out.append(idx2word[nxt])
            logits, hidden = self.forward(torch.tensor([[nxt]], device = device), hidden)

        return detokenize(out)

    def run(self):

        self.train(self.data["train"]["X"], self.data["train"]['y'])

        results = {}

        for words in self.start_phrases:
            joke = self.generate(words, self.data['word2idx'], self.data['idx2word'])
            print(words, '->', joke)
            results[' '.join(words)] = joke
        return results


