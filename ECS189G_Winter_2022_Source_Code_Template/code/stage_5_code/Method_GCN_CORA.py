import pandas as pd
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from code.base_class.method import method
import torch.nn.functional as F
import numpy as np
from code.stage_5_code.Evaluator import Evaluate
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.manual_seed(2)
np.random.seed(2)


# https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        print("input shape; ", input.shape)
        print("adj shaoe: ", adj.shape)
        support = torch.mm(input, self.weight)
        print("support.shape: ", support.shape)
        output = torch.spmm(adj, support)
        print("output_shape: ", output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Method_GCN_Cora(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.hidden = 200
        self.hidden2 = 80
        self.n_class = 7

        self.sgc_1 = GraphConvolution(1433, self.hidden)
        self.sgc_2 = GraphConvolution(self.hidden, self.hidden2)
        self.fc_1 = nn.Linear(self.hidden2, self.n_class)
        self.dropout = 0.3

    def forward(self, x, adj, bool=False, y=None, idxs=None):
        x = self.sgc_1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training) # we will be in train mode; so we want dropout to be activated
        x = F.relu(self.sgc_2(x, adj))
        if bool == True:
            print("X.shape: ", x.shape)
            self.tsnefunct(x, y, idxs)
        x = self.fc_1(x)
        print("final x shpae: ", x.shape)
        return x # going to use CE Loss, so not going to softmax.

    def train(self, X, y, adj, train_idx, test_idx):

        n_epochs = 6
        batch_size = 200
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        accuracy_evaluator = Evaluate('training evaluator', '')

        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        permutation = torch.randperm(X.size()[0])

        losses_per_epoch = []

        for epoch in range(n_epochs):
            self.training = True
            optimizer.zero_grad()

            #for i in range(0, X.size()[0], batch_size):
                #print(f"Epoch {epoch}, Batch: {i}")
                #indices = permutation[i:i+batch_size]

                #batch_x, batch_y = X[indices, :], y[indices]

                #print('batch_x.shape: ', batch_x.shape)
                #print('batch_y.shape: ', batch_y.shape)

            # Pass all of the data because of the adjacency matrix stuff
            y_pred = self.forward(X, adj)
            loss = loss_fn(y_pred[train_idx], y[train_idx])

            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            print("Running testing accuracy: ")
            losses_per_epoch.append(loss.item())
            with torch.no_grad():
                self.training = False
                if epoch == 4:
                    pred_test = self.forward(X, adj, bool=True, y=y, idxs=test_idx)
                else:
                    pred_test = self.forward(X, adj)
                accuracy_evaluator.data = {'true_y': y[test_idx], 'pred_y': torch.argmax(pred_test, dim=1)[test_idx]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', loss.item())


        plt.plot(losses_per_epoch)
        plt.title("Epochs vs Loss: CORA")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.show()
        with open("/Users/jacquelinemitchell/Documents/cora_losses.npy", "wb") as f:
            np.save(f, np.array(losses_per_epoch))


            # OKAY: debug later: basically, GCN needs the entire adj mat and thus all of the
            # node features; if we only provide certain data at idxs, like we did, we'll get a
            # mat mult error.  So basically, we need to pass all of the data, but only
            # compute the loss/acc on the desried idxs. Cool.

    def run(self):

        graph_data = self.data['graph']
        input_data = self.data['train_test']

        train_idx, test_idx = input_data['idx_train'], input_data['idx_test']
        all_inputs, all_labels = graph_data['X'], graph_data['y']
        adj = graph_data['utility']['A']

        # X_train, y_train = all_inputs, all_labels
        # X_test, y_test = all_inputs, all_inputs[test_idx]

        self.train(all_inputs, all_labels, adj, train_idx, test_idx)

    def tsnefunct(self, data, labels, idxs):

        print("data.shape: ", data.shape)

        tsne = TSNE(n_components=2, verbose=1)
        z = tsne.fit_transform(data[idxs])
        df = pd.DataFrame()
        df["y"] = labels[idxs]
        df["comp_1"] = z[:, 0]
        df["comp_2"] = z[:, 1]

        a = sns.scatterplot(x="comp_1", y="comp_2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 7),
                        data=df).set(title="T-SNE of Test Set CORA Graph Embeddings")
        plt.legend(title="Class", fontsize=7)#, labels=list(self.class_dict.values()))
        print(self.class_dict)
        plt.show()






