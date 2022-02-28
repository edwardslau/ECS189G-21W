from code.base_class.method import method
import torch
import torch.nn as nn
from code.stage_4_code.DataLoader_Generation import Dataset_Loader
import numpy as np
import torchtext
from collections import defaultdict

torch.manual_seed(2)
np.random.seed(2)

class Method_RNN_Generalization(method, nn.Module):

    def __init__(self, mName, mDescription, vocab_w_i, vocab_i_w):
        #"method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.hidden_dim = 128
        self.embedding_dim = 300
        self.num_layers = 1
        #self.bidirectional = True

        vocab_size = len(vocab_w_i)
        print("Vocab size; ", vocab_size)
        #input("...")

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

       # print("Number of words without an embedding vector: ", words_not_gloved, " out of ", len(vocab_w_i))

        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        batch_size = 300
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        print(X.shape)
        print(y.shape)

        # Create batches
        per_sentences = []
        # let's process 10 jokes worth of data at a time
        prev = 0
        prev_res = 0
        for sent in range(10, len(jokes_len), 10):
            res = sum(jokes_len[prev:sent]) - 30 # 10 jokes * - 3 for each sentence
            prev += 10
            range_tuple = (prev_res, prev_res + res)
            prev_res = res + prev_res
            per_sentences.append(range_tuple)

        print(jokes[9])
        #print(X[])

        #print(per_sentences)

        losses_per_epoch = []

        for epoch in range(1,10):
            # Every epoch, reset the hidden states and cell states of the LSTMs
            hidden_state, cell_state = self.init_state(sequence_length=46)

            for i in range(0, X.size()[0], batch_size):
            #for k in per_sentences:
                batch_x = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                # batch_x = X[k[0]:k[1]]
                # batch_y = y[k[0]:k[1]]

                #print(f"Batches: {k},:, {i+batch_size}]")

                optimizer.zero_grad()

                # forward pass
                #if i == 0:
                y_pred, h = self.forward(batch_x, None)
                # else:
                #     try:
                #         y_pred, h = self.forward(batch_x, h.data)
                #     except:
                #         y_pred, h = self.forward(batch_x, None)
                # print("y_pred.shape: ", y_pred.shape)
                # print("y_pred[0]", y_pred[0])
                # print("batch_y[0]: ", batch_y[0])
                loss = loss_fn(y_pred, batch_y)

                #hidden_state = hidden_state.detach()
                #cell_state = cell_state.detach()

                loss.backward()

                #nn.utils.clip_grad_norm_(self.parameters(), 5)
                #h.detach()
                optimizer.step()

                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            losses_per_epoch.append(loss.item())

        with open("/Users/jacquelinemitchell/Documents/rnn_generation_losses.npy", "wb") as f:
            np.save(f, np.array(losses_per_epoch))

        joke_tracker = defaultdict(int)
        total_jokes_corr = 0
        non_unique = 0

        for i in range(len(jokes)):
            joke_tracker[str(jokes[i][:3])] += 1

        for i in range(len(jokes)):
            if joke_tracker[str(jokes[i][:3])] > 1:
                non_unique += 1
                continue
            pred, real = self.generate_text(i, jokes, jokes_len, vocab_w_i, vocab_i_w)
            if pred == real:
                total_jokes_corr += 1
        print("Acc: ", total_jokes_corr / (len(jokes) - non_unique))

        #input("uhh.....")
        #self.generate_text_scratch(['when', 'do', 'you', 'call'], jokes, jokes_len, vocab_w_i, vocab_i_w)
        self.generate_text_scratch(['can', 'cats', 'fly'],  jokes, jokes_len, vocab_w_i, vocab_i_w)
        #self.generate_text_scratch(['when', 'did', 'tuna', 'meet'],  jokes, jokes_len, vocab_w_i, vocab_i_w)
        # self.generate_text(29, jokes, jokes_len, vocab_w_i, vocab_i_w)
        # self.generate_text(606, jokes, jokes_len, vocab_w_i, vocab_i_w)
        # self.generate_text(614, jokes, jokes_len, vocab_w_i, vocab_i_w)

    def generate_text_scratch(self, input_text, jokes, jokes_len, vocab_w_i, vocab_i_w):

        # assume input is passed as text:
        #starter_index = 0
        self.training = False # turn off dropout; this will cause issues with inference

        # This is necessary here, since the sequence lenght we want to input is larger.
        h = self.init_state(1)

        text_o = input_text
        #input(".....")
        #print("text OOOOO:", text_o)
        text = input_text
        length = 20

        words = text + []

        for i in range(0, length-len(text)):
            #assert True == False
            tokenized = torch.tensor([vocab_w_i[word] for word in text]).unsqueeze(0)
            #if i == 0:
            prediction, h = self.forward(tokenized, None)
            #else:
                #prediction, h = self.forward(tokenized, h)
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

        print("Seed text: ", text)
        print("Predicted Joke: ", words)
        return words, #jokes[starter_index]

    def generate_text(self, starter_index, jokes, jokes_len, vocab_w_i, vocab_i_w):

        # assume input is passed as text:
        #starter_index = 0
        text = jokes[starter_index][:3]
        length = jokes_len[starter_index]
        self.training = False # turn off dropout; this will cause issues with inference

        # This is necessary here, since the sequence lenght we want to input is larger.
        h = self.init_state(1)

        # text = ['what', 'am', 'i']
        # length = 20

        words = text + []

        for i in range(0, length-len(text)):
            #assert True == False
            tokenized = torch.tensor([vocab_w_i[word] for word in text]).unsqueeze(0)
            #if i == 0:
            prediction, h = self.forward(tokenized, None)
            #else:
                #prediction, h = self.forward(tokenized, h)
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
        return words, jokes[starter_index]

    def run(self):
        self.train(self.data['X'], self.data['y'], self.data['jokes'], self.data['jokes_len'],
                   self.data['w_i'], self.data['i_w'])
        self.generate_text_scratch(['when', 'did', 'tuna'],  self.data['jokes'], self.data['jokes_len']
                                   ,self.data['w_i'], self.data['i_w'])
        self.generate_text_scratch(['when', 'i', 'was'], self.data['jokes'], self.data['jokes_len']
                                   ,self.data['w_i'], self.data['i_w'])

if __name__ == "__main__":
    loader = Dataset_Loader()
    x, y, vocab_w_i, vocab_i_w, jokes, jokes_len = loader.load()
    mod = Method_RNN_Generalization("junk", "junk2", vocab_w_i, vocab_i_w)
    mod.train(x, y, jokes, jokes_len, vocab_w_i, vocab_i_w)


