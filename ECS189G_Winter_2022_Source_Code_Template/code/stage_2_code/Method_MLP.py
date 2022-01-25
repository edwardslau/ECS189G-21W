import torch
import torch.nn as nn
from code.base_class.method import method
import numpy as np
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_MLP(method, nn.Module):

    # Class variables
    data = None
    iterations = 3000
    batch_size = 100
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        """Initialization"""
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc_1 = nn.Linear(784, 100)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(100, 100)
        self.act_2 = nn.ReLU()
        self.fc_3 = nn.Linear(100, 10) # 10 Output classes in MNIST Dataset
        #self.act_prob = nn.Softmax(dim=1)

    def forward(self, input_x):
        """Propagate Forward

        :param input_x: arguments to be put through a forward pass

        """
        out = self.fc_1(input_x)
        out = self.act_1(out)
        out = self.fc_2(out)
        out = self.act_2(out)
        out = self.fc_3(out)
        #output_predictions = self.act_prob(input_x)

        return out

    def train(self, X, y):
        """
        :param X: input data (images)
        :param y: input data labels (0-9 digits)
        """

        # Selection of epochs dependent on how much training data.
        n_epochs = self.iterations / (len(X) / self.batch_size)

        # Optimizer and Loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_criterion = torch.nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        print(f"Running for {n_epochs} epochs...")

        iter = 0

        for epoch in range(int(n_epochs)):
            # This is set up so that we can run mini-batches one at time, although
            # this assumes that the train data was set up to be enumerated in the correct way.

            # Mini-batching code credit: https://stackoverflow.com/questions/45113245/how-to-get
            # -mini-batches-in-pytorch-in-a-clean-and-efficient-way
            permutation = torch.randperm(X.size()[0])

            for i in range(0, X.size()[0], self.batch_size):
                # Do this for each batch, as PyTorch accumulates gradients
                optimizer.zero_grad()

                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = X[indices], y[indices]

                # Forward pass
                outputs_pred = self.forward(batch_x) # again, may need to edit how train_d is passed depending on dtype
                loss = loss_criterion(outputs_pred, batch_y)
                # Update params
                loss.backward() # back-prop the errors
                optimizer.step() # Update via gradient descent

                iter += 1

                if iter % 500 == 0:
                    # evaluate how training process is going... WIP.
                    print("Current Progress::....")
                    #print(batch_y.shape)
                    current_full_outputs = self.forward(X)
                    accuracy_evaluator.data = {'true_y': y, 'pred_y': current_full_outputs.max(1)[1]}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', loss.item())


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

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}










