from torch_geometric.nn import GCNConv
from code.base_class.method import method
from code.stage_5_code.Evaluator import Evaluate
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd


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
        #print("input shape; ", input.shape)
        #print("adj shaoe: ", adj.shape)
        support = torch.mm(input, self.weight)
        #print("support.shape: ", support.shape)
        output = torch.spmm(adj, support)
        #print("output_shape: ", output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Method_GCN_PUBMED(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.hidden = 300
        self.hidden2 = 75
        self.n_class = 3

        self.sgc_1 = GraphConvolution(500, self.hidden)
        self.sgc_2 = GraphConvolution(self.hidden, self.hidden2)
        self.sgc_3 = GraphConvolution(self.hidden2, self.hidden2)
        self.dropout = 0.3
        self.fc_1 = nn.Linear(self.hidden2, self.n_class)

    def forward(self, x, adj, bool=False, y=None, idx=None):
        x = self.sgc_1(x, adj)
        x = F.relu(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.sgc_2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sgc_3(x, adj)
        x = F.relu(x)
        if bool == True:
            self.tsnefunct(x, y, idx)
        x = self.fc_1(x)
        return x

    def train(self, X, y, adj, train_idx, test_idx):
        n_epochs = 17
        batch_size = 200
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        accuracy_evaluator = Evaluate('training evaluator', '')

        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        permutation = torch.randperm(X.size()[0])

        losses_per_epoch = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # for i in range(0, X.size()[0], batch_size):
            # print(f"Epoch {epoch}, Batch: {i}")
            # indices = permutation[i:i+batch_size]

            # batch_x, batch_y = X[indices, :], y[indices]

            # print('batch_x.shape: ', batch_x.shape)
            # print('batch_y.shape: ', batch_y.shape)

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
                if epoch == n_epochs - 1:
                    pred_test = self.forward(X, adj, bool=True, y=y, idx=test_idx)
                else:
                    pred_test = self.forward(X, adj)

                print("Pred test shape: ", pred_test.shape)
                accuracy_evaluator.data = {'true_y': y[test_idx], 'pred_y': torch.argmax(pred_test, dim=1)[test_idx]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', loss.item())

            # OKAY: debug later: basically, GCN needs the entire adj mat and thus all of the
            # node features; if we only provide certain data at idxs, like we did, we'll get a
            # mat mult error.  So basically, we need to pass all of the data, but only
            # compute the loss/acc on the desried idxs. Cool.

        plt.plot(losses_per_epoch)
        plt.title("Epochs vs Loss: PubMed")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.show()

    def run(self):
        graph_data = self.data['graph']
        input_data = self.data['train_test']

        train_idx, test_idx = input_data['idx_train'], input_data['idx_test']
        all_inputs, all_labels = graph_data['X'], graph_data['y']
        adj = graph_data['utility']['A']

        print("Length of train_idx: ", len(train_idx))
        print("Length of test_udx: ", len(test_idx))

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
                        palette=sns.color_palette("hls", 3),
                        data=df).set(title="T-SNE of Test Set PubMed Graph Embeddings")
        plt.legend(title="Class", fontsize=7)#, labels=list(self.class_dict.values()))
        print(self.class_dict)
        plt.show()






