from code.base_class.method import method
from code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader
from code.stage_3_code.Evaluator import Evaluate_Accuracy
from code.stage_3_code.Evaluator_All import Evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

class Method_CNN_CIFAR(method, nn.Module):
    data = None
    learning_rate = 1e-04
    epochs = 30

    # https://www.sciencedirect.com/science/article/pii/S2405959519303455#:~:text=The%20optimizer%20used%20for%20both,with%20the%2016%20batch%20size.
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Preserve the size of the convolution since the image is very small
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same")
        self.act_1 = nn.ReLU()
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool_1 = nn.AvgPool2d(2, 2)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm2d(128)
        self.pool_2 = nn.AvgPool2d(2, 2)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.act_3 = nn.ReLU()
        self.bn_3 = nn.BatchNorm2d(256)
        self.pool_3 = nn.MaxPool2d(2, 2)
        #output size is 20x16x16 (PyTorch uses a NCHW scheme).
        # Batch normalization before passing to output
        # We now do the MLP component
        self.flat = nn.Flatten() # the output will now be (num_samples, 20 * 16 * 16)
        self.fc_1 = nn.Linear(4096, 512) # reduce by a factor of 4
        self.act_4 = nn.ReLU()
        self.drop_1 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(512, 10)


    def forward(self, X):

        output = self.conv_1(X)
        output = self.act_1(output)
        output = self.bn_1(output)
        output = self.pool_1(output)
        output = self.conv_2(output)
        output = self.act_2(output)
        output = self.bn_2(output)
        output = self.pool_2(output)
        output = self.conv_3(output)
        output = self.act_3(output)
        output = self.bn_3(output)
        output = self.pool_3(output)

        output = self.flat(output)
        output = self.fc_1(output)
        output = self.act_4(output)
        output = self.drop_1(output)
        output = self.fc_2(output)
        return output

    def train(self, X, y):


        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(self.parameters())
        accuracy_evaluator = Evaluate('training evaluator', '')
        batch_size = 1024

        # Prep data
        X = torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2)
        y = torch.LongTensor(np.array(y))
        permutation = torch.randperm(X.size()[0])

        losses_total = []

        # We will NOT attempt Full-batch SGD again, too large to fit into memory at once
        # Credit for minibatching code to:
        # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
        for epoch in range(1, self.epochs + 1):
            print("Epoch: ", epoch)

            for i in range(0, X.size()[0], batch_size):
                self.training = True
                optimizer.zero_grad()

                #print("selecting indices")
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], y[indices]
                batch_x, batch_y = batch_x.float().requires_grad_(), batch_y
                #print(batch_x.requires_grad, batch_y.requires_grad)
                #print("batch_x shape: ", batch_x.shape)

                #print("Running inference...")
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

            if epoch == 50:
                print("HIIII")
                with torch.no_grad():
                    self.training = False
                    X_test = torch.FloatTensor(np.array(self.data['test']['X'])).permute(0, 3, 1, 2)
                    pred_test = self.forward(X_test)
                    accuracy_evaluator.data = {'true_y' : self.data['test']['y'], 'pred_y' : torch.argmax(pred_test, dim=1)}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', losses.item())

            losses_total.append(losses.item())

        with open("/Users/jacquelinemitchell/Documents/cifar_10_losses.npy", "wb") as f:
            np.save(f, np.array(losses_total))

    def test(self, X):
        y_pred = self.forward(torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2))
        return torch.argmax(y_pred, axis=1)

    def run(self):
        # data_obj = Dataset_Loader('cifar', '')
        # data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        # data_obj.dataset_source_file_name = 'CIFAR'
        #
        # data = data_obj.load()
        #
        # self.data = {'train': {'X': data['X_train'], 'y': data['y_train']},
        #              'test': {'X': data['X_test'], 'y': data['y_test']}}

        print(type(self.data['train']['X']))
        self.train(self.data['train']['X'], self.data['train']['y'])
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}


# if __name__ == "__main__":
#     a = Method_CNN_CIFAR("cifar", '')
#     a.run()








