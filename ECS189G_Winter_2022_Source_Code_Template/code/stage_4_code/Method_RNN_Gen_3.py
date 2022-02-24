from code.base_class.method import method
import torch
import torch.nn as nn
from code.stage_4_code.DataLoader_Generation import Dataset_Loader
import numpy as np
import torchtext

torch.manual_seed(2)
np.random.seed(2)

class Method_RNN_Generalization(method, nn.Module):

    def __init__(self, mName, mDescription, vocab_w_i, vocab_i_w):
        #"method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.hidden_dim = 300
        self.embedding_dim = 256
        self.num_layers = 3

        vocab_size = len(vocab_w_i)

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.fc_1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x) # returns: [batch_size, sequence_length, embedding_dim]
        output, hidden_out = self.lstm(embedded, hidden) # returns output: [batch_size, sequence_length, hidden_dim]
        outs = self.fc_1(output)
        return outs, hidden_out

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_dim),
                torch.zeros(self.num_layers, sequence_length, self.hidden_dim))

    def train(self, X, y, jokes, jokes_len, vocab_w_i, vocab_i_w):

        # words_not_gloved = 0
        # glove = torchtext.vocab.GloVe(name="6B", dim=100)
        # for word in list(vocab_w_i.keys()):
        #     try:
        #         self.embedding.weight.data[vocab_w_i[word]] = glove.vectors[glove.stoi[word]]
        #     except:
        #         self.embedding.weight.data[vocab_w_i[word]] = torch.zeros(self.embedding_dim)
        #         words_not_gloved += 1
        #
        # print("Number of words without an embedding vector: ", words_not_gloved, " out of ", len(vocab_w_i))

        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        batch_size = 300
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        print(X.shape)
        print(y.shape)

        for epoch in range(1,100):
            # Every epoch, reset the hidden states and cell states of the LSTMs
            hidden_state, cell_state = self.init_state(sequence_length=3)

            for i in range(0, X.size()[0], batch_size):
                batch_x = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                optimizer.zero_grad()

                # forward pass
                y_pred, (hidden_state, cell_state) = self.forward(batch_x, (hidden_state, cell_state))
                loss = loss_fn(y_pred.transpose(1, 2), batch_y)

                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

        self.generate_text(4, jokes, jokes_len, vocab_w_i, vocab_i_w)

    def generate_text(self, starter_index, jokes, jokes_len, vocab_w_i, vocab_i_w):

        # assume input is passed as text:
        #starter_index = 0
        text = jokes[starter_index][:3]
        length = jokes_len[starter_index]
        self.training = False # turn off dropout; this will cause issues with inference

        # This is necessary here, since the sequence lenght we want to input is larger.
        h = self.init_state(len(text))

        words = text + []

        # 3 because we want to generate the entire joke given 3 input words
        for i in range(0, length-len(text)):
            tokenized = torch.tensor([vocab_w_i[word] for word in text]).unsqueeze(0)
            prediction, h = self.forward(tokenized, h)

            print("prediction shape: ", prediction.shape)

            last_word_logits = prediction[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            #largest_prob = np.argmax(p)
            #print(largest_prob)
            #word_index = np.random.choice(len(last_word_logits), p=p)
            word_index = np.argmax(p)
            words.append(vocab_i_w[word_index])
            text = text[1:] + [vocab_i_w[word_index]]

        print("Seed text: ", jokes[starter_index][:3])
        print("Real joke: ", jokes[starter_index])
        print("Predicted Joke: ", words)
        return words

if __name__ == "__main__":
    loader = Dataset_Loader()
    x, y, vocab_w_i, vocab_i_w, jokes, jokes_len = loader.load()
    mod = Method_RNN_Generalization("junk", "junk2", vocab_w_i, vocab_i_w)
    mod.train(x, y, jokes, jokes_len, vocab_w_i, vocab_i_w)


