from code.base_class.method import method
from code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader
from code.stage_3_code.Evaluator import Evaluate_Accuracy
import torch
import torch.nn as nn
import numpy as np
import os

class Method_CNN_MNIST(method, nn.Module):
    data = None
    learning_rate = 1e-03
    epochs = 10

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=40, kernel_size=3)
        self.act_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        self.fc_1 = nn.Linear(1000, 10)

    def forward(self, X):

        out = self.conv_1(X)
        out = self.act_1(out)
        out = self.pool_1(out)
        out = self.conv_2(out)
        out = self.act_2(out)
        out = self.pool_2(out)


        # this gives us a 2D tensor (num_images, channels * im_dim)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)

        return out

    def train(self, X, y):

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        batch_size = 500

        # Prep data
        X = X.unsqueeze(-1)
        print(X.shape)
        X = torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2)
        print(X.shape)
        y = torch.LongTensor(np.array(y))
        permutation = torch.randperm(X.size()[0])

        losses_total = []

        # We will NOT attempt Full-batch SGD again, too large to fit into memory at once
        # Credit for minibatching code to:
        # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
        for epoch in range(1, self.epochs + 1):
            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], y[indices]
                batch_x, batch_y = batch_x.float().requires_grad_(), batch_y

                y_pred = self.forward(batch_x)
                #print("Done running inference")
                #print("Calculating losses...")
                losses = loss_function(y_pred, batch_y)
                #print("Done calcualting losses")

                #print("Gradients starting...")
                losses.backward()
                #print("Gradients Done...")
                #print("optimizer starting...")
                optimizer.step()
                #print("optimizer ending..")

            if True:
                print("HIIII")
                with torch.no_grad():
                    X_test = torch.FloatTensor(np.array(self.data['test']['X'])).unsqueeze(-1).permute(0, 3, 1, 2)
                    pred_test = self.forward(X_test)
                    accuracy_evaluator.data = {'true_y' : self.data['test']['y'], 'pred_y' : torch.argmax(pred_test, dim=1)}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', losses.item())

            losses_total.append(losses.item())

            with open("/Users/jacquelinemitchell/Documents/mnist_losses.npy", "wb") as f:
                np.save(f, np.array(losses_total))

    def test(self, X):
        X = X.unsqueeze(-1).permute(0, 3, 1, 2)
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return torch.argmax(y_pred, axis=1)

    def run(self):
        data_obj = Dataset_Loader('mnist', '')
        data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        data_obj.dataset_source_file_name = 'MNIST'

        data = data_obj.load()

        self.data = {'train': {'X': data['X_train'], 'y': data['y_train']},
                     'test': {'X': data['X_test'], 'y': data['y_test']}}

        print(type(self.data['train']['X']))
        self.train(self.data['train']['X'], self.data['train']['y'])
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

if __name__ == "__main__":
    a = Method_CNN_MNIST("mnist", '')
    a.run()



