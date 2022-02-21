from code.base_class.method import method
from code.stage_4_code.DataLoader_Class_2 import Dataset_Loader
import torch
import torch.nn as nn
import numpy as np
import torchtext

np.random.seed(2)
torch.manual_seed(2)


class Method_RNN_Classification(method, nn.Module):
    data = None
    learning_rate = 1e-03
    vocab_w_i = None

    def __init__(self, mName, mDescription, vocab_size):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding_dim = 100
        self.hidden_dim = 200
        self.output_dim = 1
        self.sequence_length = 500
        self.vocab_size = vocab_size
        self.num_layers = 2

        # output of embedding: [sequence_length, batch_size, embed_dim]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # output of RNN:
        # output: [sequence_length, batch_size, hidden_dim]
        # h_n: [num_layers, batch_size, hidden_dim] (Contains final hidden state for each
        # element in a batch)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        # it is hidden_dim * sequence length because (we have sequence-length
        # iterations of the model over time and hidden_dim number of outputs
        # for each of those iterations.)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        #self.sigmoid = torch.nn.Sigmoid()
        self.word_dict = None

    def forward(self, text, hid):

        # embedded = [sent len, batch size, emb dim]
        embedded = self.embedding(text)

        # hidden returns to you the value at the very last timestep
        output, hidden_out = self.rnn(embedded, hid)  # output: [batch_size, sent len, output dim]

        # hn = hidden.view(-1, 200)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        # print("outputshape: ", output.size())
        # print("hiddenshape: ", hidden.size())
        # assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        # print("outputshape: ", output.size())
        batch_size = output.shape[0] # record batch size

        output = output.contiguous().view(-1, self.hidden_dim)

        #output = output.contiguous().view(-1, self.hidden_dim)
        output = self.dropout(output) # just some dropout
        o = self.fc(output) # results in [batch_size, sent_len, 1]
        o = o.view(batch_size, -1)

        # print("Hidden squeeze: ", hidden.squeeze(0).shape)
        # o = self.fc(hidden.squeeze(0))

        # print("o-shape", o.shape)
        # o[:, -1] means we return the last time step only
        return o[:, -1], hidden_out

    def train(self, X, y, X_test, y_test, vocab):

        # Before we begin training, we load our vectors!
        words_not_gloved = 0
        glove = torchtext.vocab.GloVe(name="6B", dim=100)
        for word in list(vocab.keys()):
            try:
                self.embedding.weight.data[vocab[word]] = glove.vectors[glove.stoi[word]]
            except:
                self.embedding.weight.data[vocab[word]] = torch.zeros(self.embedding_dim)
                words_not_gloved += 1

        print("Number of words without an embedding vector: ", words_not_gloved, " out of ", len(vocab))

        # make the padding and unknown indexes start at 0
        self.embedding.weight.data[0] = torch.zeros(self.embedding_dim)
        self.embedding.weight.data[1] = torch.zeros(self.embedding_dim)

        X = torch.LongTensor(X) # X.size = torch.Size is (200,500)
        y = torch.FloatTensor(y)

        print(X.size())
        print(y.size())

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        batch_size = 64#1000

        losses_per_epoch = []

        epochs = 5
        for epoch in range(epochs):
            print("Epoch: ", epoch)

            self.training = True # activate dropout
            for i in range(0, X.size()[0], batch_size):
                print("i: ", i)
                optimizer.zero_grad()

                # print("i:i + batch_size: ", i, ":", i + batch_size)

                batch_x, batch_y = X[i:i + batch_size, :], y[i:i+batch_size]

                h0 = torch.zeros((self.num_layers, batch_x.size()[0], self.hidden_dim))
                c0 = torch.zeros((self.num_layers, batch_x.size()[0], self.hidden_dim))

                # print(batch_x[:, 0])
                # print([vocab[i] for i in batch_x[:, 0]])

                # print("requires grad? ", batch_x.requires_grad)
                # print("batch_x shape: ", batch_x.shape)
                # print("batch_y shape: ", batch_y.shape)

                output, hidden_out = self.forward(batch_x, (h0, c0))
                output = output.squeeze(-1)
                # print("Output shapel: ", output.shape)
                # print("Output shape: ", batch_y.shape)
                loss = loss_fn(output, batch_y)
                # print("Loss: ", loss.item())
                loss.backward()

                # Addresses exploding gradients...
                #nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()

            if True:
                print("HIIII")
                print(loss.item())
                with torch.no_grad():
                    self.training = False
                    print(f"self.training status: {self.training}")
                    X_test = torch.LongTensor(X_test)
                    #X_test = X_test[:X_test.size()[0]//2]
                    #y_test = y_test[:len(y_test) // 2]

                    total_corr = 0

                    for j in range(0, X_test.size()[0], batch_size):

                        test_batch_x, test_batch_y = X_test[j:j+batch_size,:], y_test[j:j+batch_size]

                        # (num_layers, batch_size, hidden_dimension)
                        h0 = torch.zeros((self.num_layers, test_batch_x.size()[0], self.hidden_dim))
                        c0 = torch.zeros((self.num_layers, test_batch_x.size()[0], self.hidden_dim))

                        pred_test, h_out = self.forward(test_batch_x, (h0, c0))
                        pred_test = pred_test.squeeze(-1)
                        # print(pred_test)
                        #print(pred_test.shape)
                        rounded_preds = torch.round(torch.sigmoid(pred_test))
                        # print(rounded_preds)
                        # print(rounded_preds.shape)
                        correct = np.array(rounded_preds) == np.array(test_batch_y) # convert into float for division
                        #print(correct)
                        acc = sum(correct) / len(correct)
                        total_corr += sum(correct)
                        print('Epoch:', epoch, 'Batch:', j, '  Accuracy:', acc, 'Loss:', loss.item())
                    print("TOTAL EPOCH TEST ACCURACY: ", total_corr / X_test.size()[0])
                losses_per_epoch.append(loss.item())

        with open("/Users/jacquelinemitchell/Documents/IMDB_classification_losses.npy", "wb") as f:
            np.save(f, np.array(losses_per_epoch))

    def test(self, X):

        if self.training == False:
            self.training = True

        X = torch.LongTensor(X)
        batch_size = 64

        predictions = np.array([])

        with torch.no_grad():
            for i in range(0, X.size()[0], batch_size):

                batch_x = X[i:i + batch_size, :]

                h0 = torch.zeros((1, batch_x.size()[0], self.hidden_dim))
                c0 = torch.zeros((1, batch_x.size()[0], self.hidden_dim))

                pred_test, h_out = self.forward(batch_x, (h0, c0))
                pred_test = pred_test.squeeze(-1)
                rounded_preds = np.array(torch.round(torch.sigmoid(pred_test)))

                predictions = np.append(predictions, rounded_preds)

        return predictions

    def run(self):
        self.train(self.data['train']['X'], self.data['train']['y'],
                   self.data['test']['X'], self.data['test']['y'], self.vocab_w_i)

        a = self.test(self.data['test']['X'])
        return {'pred_y' : a, 'true_y' : self.data['test']['y']}


if __name__ == "__main__":
    DL = Dataset_Loader()
    data_dict, vocab = DL.load()
    mod = Method_RNN_Classification(mName="HI", mDescription="NO",
                                    vocab_size=1001)#len(vocab) + 2)  # Note: the reason why we add
    # + 2 here is because we cannot forget about our dummy variables (0 : pad, 1 : unk)
    mod.train(data_dict["X_train"], data_dict["y_train"], data_dict["X_test"], data_dict["y_test"], vocab)