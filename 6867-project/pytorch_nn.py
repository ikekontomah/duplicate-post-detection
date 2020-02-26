import torch
from torch.autograd import Variable
from torch.utils.data import *

class NeuralNetwork:
    def __init__(self, in_size, hidden_size, out_size):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_size*4),
            torch.nn.ReLU(),
            torch.nn.Linear(out_size*4, out_size),
            torch.nn.Softmax()
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)

    def train(self,X_train,Y_train):
        epochs = 100
        trainset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
        trainloader = DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
        for e in range(epochs):
            for i, data in enumerate(trainloader, 0):
                # Make data type Variable
                x, y = data
                x, y = Variable(x), Variable(y)

                self.optimizer.zero_grad()

                # Forward Pass & Calculate Loss
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred,y)

                # Backwards pass
                loss.backward()
                self.optimizer.step()
            train_score = self.score(X_train, Y_train)
            print("Train Accuracy: %s" % train_score)
        return train_score

    def score(self,X,Y):
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long())
        loader = DataLoader(dataset, batch_size=32,
                                          shuffle=True, num_workers=2)
        correct = 0
        total = 0
        for data in loader:
            x, y = data
            outputs = self.model(Variable(x))
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()

        return 100*(float(correct)/total)

