import torch
import torch.nn as nn

class MLP(nn.Module):

    # Class variables
    data = None
    iterations = 3000
    batch_size = 100
    learning_rate = 1e-1

    def __init__(self):
        """Initialization"""
        super(MLP, self).__init__()

        self.fc_1 = nn.Linear(784, 100)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(100, 100)
        self.act_2 = nn.ReLU()
        self.fc_3 = nn.Linear(100, 10) # 10 Output classes in MNIST Dataset
        self.act_prob = nn.Softmax(dim=1)

    def forward(self, input_x):
        """Propagate Forward

        :param input_x: arguments to be put through a forward pass

        """
        input_x = self.fc_1(input_x)
        input_x = self.act_1(input_x)
        input_x = self.fc_2(input_x)
        input_x = self.act_2(input_x)
        input_x = self.fc_3(input_x)
        output_predictions = self.act_prob(input_x)

        return output_predictions

    def train(self, X, y):
        """
        :param X: input data (images)
        :param y: input data labels (0-9 digits)
        """

        # Selection of epochs dependent on how much training data.
        n_epochs = self.iterations / (len(X) / self.batch_size)

        # Optimizer and Loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_criterion = torch.nn.NLLLoss() # useful for categorical classification tasks (just arbitrary for now)

        # Set up data loading process
        train_loader = None # needs to be set up
        test_loader = None

        for epoch in range(n_epochs):
            # This is set up so that we can run mini-batches one at time, although
            # this assumes that the train data was set up to be enumerated in the correct way.
            for i, (train_d, labels) in enumerate(train_loader):

                # May need to change depending on how the data is loaded
                # I assume for now that train_d passes us tensors.
                train_d.requires_grad()

                # Do this for each batch, as PyTorch accumulates gradients
                optimizer.zero_grad()

                # Forward pass
                outputs_pred = self.forward(train_d) # again, may need to edit how train_d is passed depending on dtype

                loss = loss_criterion(outputs_pred, labels)

                loss.backward() # back-prop the errors
                optimizer.step() # Update via gradient descent

                if epoch % 50 == 0:
                    # evaluate how training process is going... WIP.
                    print("Current Progress::....")

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
        self.train(train_data, train_data_labels)
        print('Test Results')
        pred_y = self.test(test_data)
        #return {'pred_y': pred_y, 'true_y': self.data['test']['y']}










