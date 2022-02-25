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

        self.hidden_dim = 128
        self.embedding_dim = 300
        self.num_layers = 1

        vocab_size = len(vocab_w_i)
        print("Vocab size; ", vocab_size)

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc_1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x) # returns: [batch_size, sequence_length, embedding_dim]
        #print("Embedding shpae: ", embedded.shape)
        output, hidden_out = self.lstm(embedded, hidden) # returns output: [batch_size, sequence_length, hidden_dim]
        #print("Output shape: ", output.shape)
        #print("Hidden_out.shape: ", [h.shape for h in hidden_out])
        #outs = self.dropout(output)
        outs = self.fc_1(output)
        #print("outs.shape: ", outs.shape)
#        outs = torch.argmax(outs, dim=2)
        outs = outs[:,-1,:] # only retrieve output at the last time step
        #print("outs.shape: ", outs.shape)
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

        for epoch in range(1,10):
            # Every epoch, reset the hidden states and cell states of the LSTMs
            hidden_state, cell_state = self.init_state(sequence_length=46)

            for i in range(0, X.size()[0], batch_size):
                batch_x = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                print(f"Batches: {i},:, {i+batch_size}]")

                optimizer.zero_grad()

                # forward pass
                y_pred, _ = self.forward(batch_x, None)
                loss = loss_fn(y_pred, batch_y)

                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

        self.generate_text(20, jokes, jokes_len, vocab_w_i, vocab_i_w)
        self.generate_text(29, jokes, jokes_len, vocab_w_i, vocab_i_w)
        self.generate_text(606, jokes, jokes_len, vocab_w_i, vocab_i_w)
        self.generate_text(614, jokes, jokes_len, vocab_w_i, vocab_i_w)

    def generate_text(self, starter_index, jokes, jokes_len, vocab_w_i, vocab_i_w):

        # assume input is passed as text:
        #starter_index = 0
        text = jokes[starter_index][:3]
        length = jokes_len[starter_index]
        self.training = False # turn off dropout; this will cause issues with inference

        # This is necessary here, since the sequence lenght we want to input is larger.
        h = self.init_state(1)

        words = text + []

        for i in range(0, length-len(text)):
            #assert True == False
            tokenized = torch.tensor([vocab_w_i[word] for word in text]).unsqueeze(0)
            #if i == 0:
            prediction, h = self.forward(tokenized, None)
            # else:
            #     prediction, h = self.forward(tokenized, h)
            prediction = prediction.squeeze(0)

            print(prediction.shape)
            next_word = torch.argmax(prediction).item()
            print("next predicted word: ", vocab_i_w[next_word])
            words.append(vocab_i_w[next_word])
            text = text[1:] + [vocab_i_w[next_word]]
            print("Next text: ", text)

        # # 3 because we want to generate the entire joke given 3 input words
        # for i in range(0, length-len(text)):
        #     tokenized = torch.tensor([vocab_w_i[word] for word in text]).unsqueeze(0)
        #     prediction, h = self.forward(tokenized, h)
        #
        #     print("prediction shape: ", prediction.shape)
        #
        #     last_word_logits = prediction[0][-1]
        #     p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        #     #largest_prob = np.argmax(p)
        #     #print(largest_prob)
        #     #word_index = np.random.choice(len(last_word_logits), p=p)
        #     word_index = np.argmax(p)
        #     words.append(vocab_i_w[word_index])
        #     text = text[1:] + [vocab_i_w[word_index]]

        print("Seed text: ", jokes[starter_index][:3])
        print("Real joke: ", jokes[starter_index])
        print("Predicted Joke: ", words)
        return words

if __name__ == "__main__":
    loader = Dataset_Loader()
    x, y, vocab_w_i, vocab_i_w, jokes, jokes_len = loader.load()
    mod = Method_RNN_Generalization("junk", "junk2", vocab_w_i, vocab_i_w)
    mod.train(x, y, jokes, jokes_len, vocab_w_i, vocab_i_w)


