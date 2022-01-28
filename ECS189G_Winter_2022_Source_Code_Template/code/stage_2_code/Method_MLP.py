import torch
import torch.nn as nn
from code.base_class.method import method
import numpy as np
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import pandas as pd

class Method_MLP(method, nn.Module):

    # Class variables
    data = None
    iterations = 60000
    batch_size = 100
    learning_rate = 1e-3 # 5e-1

    def __init__(self, mName, mDescription):
        """Initialization"""
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc_1 = nn.Linear(784, 100)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(100, 100)
        self.act_2 = nn.ReLU()
        self.fc_3 = nn.Linear(100, 10) # 10 Output classes in MNIST Dataset

    def forward(self, input_x):
        """Propagate Forward

        :param input_x: arguments to be put through a forward pass

        """
        out = self.fc_1(input_x)
        out = self.act_1(out)
        out = self.fc_2(out)
        out = self.act_2(out)
        out = self.fc_3(out)
        # output_predictions = self.act_prob(input_x)

        return out

    def train(self, X, y):
        """
        :param X: input data (images)
        :param y: input data labels (0-9 digits)
        """

        # Selection of epochs dependent on how much training data.
        n_epochs = 300#self.iterations / (len(X) / self.batch_size)

        # Optimizer and Loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        #X = torch.FloatTensor(X)
        #y = torch.LongTensor(y)

        print(f"Running for {n_epochs} epochs...")
        df = pd.DataFrame(columns = ['Epochs', 'Loss'])

        #iter = 0

        for epoch in range(int(n_epochs)):
            optimizer.zero_grad()

            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_criterion(y_pred, y_true)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()
            graph_data = [epoch, train_loss.item()]
            df.loc[len(df)] = graph_data

            if epoch % 10 == 0:
                # evaluate how training process is going... WIP.
                print("Current Progress::....")
                accuracy_evaluator.data = {'true_y': y_true.detach().numpy(), 'pred_y': y_pred.max(1)[1].detach().numpy()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

        df.to_csv('loss_data.csv')


    def test(self, X):
        """
        :param X: testing data
        :return:
        """
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return torch.argmax(y_pred, axis=1)

    def run(self):
        """
        NOTE: JUST TEMPLATED.  Not meant to be functional yet.
        """
        print('Beginning Training Process...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('Test Results')
        pred_y = self.test(self.data['test']['X'])

        print(pred_y)
        print(self.data["test"]['y'])

        print("Testing Set Accuracy: ", sum(np.array(pred_y) == np.array(self.data["test"]['y'])) / len(pred_y))
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': self.data["test"]['y'], 'pred_y': pred_y}
        print(accuracy_evaluator.evaluate())

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}










