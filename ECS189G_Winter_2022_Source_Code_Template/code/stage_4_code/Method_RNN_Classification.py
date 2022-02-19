from code.base_class.method import method
from code.stage_4_code.DataLoader_Classification import Dataset_Loader
import torch
import torch.nn as nn
import numpy as np

np.random.seed(2)
torch.manual_seed(2)


class Method_RNN_Classification(method, nn.Module):
    data = None
    learning_rate = 1e-03

    def __init__(self, mName, mDescription, vocab_size):
        #method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding_dim = 100
        self.hidden_dim = 250
        self.output_dim = 1
        self.sequence_length = 200

        # Step 1: Embed our dimensions! vocab --> each word embedded into a vector of 100
        # output: [batch_size, sequence_length, embedding_dim]
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        # Step 2: LSTM it
        # we input embedding_dim, assume 1 item in batch, then we have the LSTM neuron
        # process one word (one embedding dim) at a time
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, num_layers=1)

        # Step 3: Squash the output
        # We will not include sigmoid activation since we are using BCEWithLogits(), which
        # already computes the sigmoid for us.
        self.fc = nn.Linear(self.sequence_length * self.hidden_dim, 1)

    def forward(self, text):

        batch_size = text.size()[0] # batch first style

        # Step 1, embed the works
        # output: [batch_size, sequence length (# words), embedding_dim]
        embedded = self.embedding(text)
        #print("embedded_output: ", embedded.size())

        # Step 2: we input to LSMT:
        # output: [batch_size, sequence_length, hidden_dim]
        output, _ = self.lstm(embedded)
        #print("lstm output: ", output.size())

        # Step 3: we output to the linear layer
        output = output.contiguous().view(batch_size, -1)
        #print("output reshaped size: ", output.size())

        output = self.fc(output)
        #print("output after fc: ", output.size())

        # Step 4: return output; we don't sigmoid because we use BCEWithLogits.
        return output

    def train(self, X, y, X_test, y_test, vocab):

        print(X.shape) # [500 examples, 200 sequence length]
        print(y.shape) # [500]

        X = torch.LongTensor(X)
        y = torch.LongTensor(y)

        # When this worked nicely it was SGD
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        batch_size = 64

        epochs = 5
        for epoch in range(epochs):
            print("Epoch: ", epoch)

            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()

                # Get the appropriate batching chunks
                batch_x, batch_y = X[i:i + batch_size, :], y[i:i + batch_size]

                # forward propagate
                # we will get output: [batch_size, 1]
                output = self.forward(batch_x)

                # We want just [batch_size] so we can compute loss, so we squeeze at the last dimension.
                output = output.squeeze(-1)
                loss = loss_fn(output, batch_y.float())
                loss.backward()

                # Addresses exploding gradients...
                nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()

            if True:
                print("HIIII")
                print(loss.item())
                # with torch.no_grad():
                #     test_data = torch.LongTensor(X_test)
                #
                #     num_corr = 0
                #     for i in range(0, test_data.size()[0], batch_size):
                #         test_batch_x = X_test[i:i+batch_size, :]
                #         test_batch_y = y_test[i:i+batch_size]
                #
                #         pred_test = self.forward(test_data)
                #         pred_test = pred_test.squeeze(-1)
                #         rounded_preds = torch.round(torch.sigmoid(pred_test))
                #
                #         correct = (np.array(rounded_preds) == np.array(y_test))  # convert into float for division
                #         num_corr += correct.sum()
                #
                #     print('Epoch:', epoch, 'Accuracy:', num_corr / test_data.size()[0] , 'Loss:', loss.item())


if __name__ == "__main__":
    DL = Dataset_Loader()
    data_dict, vocab = DL.load()
    mod = Method_RNN_Classification(mName="HI", mDescription="NO", vocab_size=len(vocab) + 2) # Note: the reason why we add
    # + 2 here is because we cannot forget about our dummy variables (0 : pad, 1 : unk)
    mod.train(data_dict["X_train"], data_dict["y_train"], data_dict["X_test"], data_dict["y_test"], vocab)



