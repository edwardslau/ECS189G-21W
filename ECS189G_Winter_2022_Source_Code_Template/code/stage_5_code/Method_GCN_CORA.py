import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from code.base_class.method import method
import torch.nn.functional as F
import numpy as np

torch.manual_seed(2)
np.random.seed(2)

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
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
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

        self.hidden = 500
        self.n_class = 7

        self.sgc_1 = GraphConvolution(1433, self.hidden)
        self.sgc_2 = GraphConvolution(self.hidden, self.n_class)
        self.dropout = 0.3

    def forward(self, x, adj):
        x = self.sgc_1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training) # we will be in train mode; so we want dropout to be activated
        x = self.sgc_2(x, adj)
        return x # going to use CE Loss, so not going to softmax.

    def train(self, X, y, adj):

        n_epochs = 5
        batch_size = 200
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        permutation = torch.randperm(X.size()[0])

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            #for i in range(0, X.size()[0], batch_size):
                #print(f"Epoch {epoch}, Batch: {i}")
                #indices = permutation[i:i+batch_size]

                #batch_x, batch_y = X[indices, :], y[indices]

                #print('batch_x.shape: ', batch_x.shape)
                #print('batch_y.shape: ', batch_y.shape)

            y_pred = self.forward(X, adj)
            loss = loss_fn(y_pred, X)

            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

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

        X_train, y_train = all_inputs[train_idx, :], all_labels[train_idx]
        X_test, y_test = all_inputs[test_idx, :], all_inputs[test_idx]

        self.train(X_train, y_train, adj)






