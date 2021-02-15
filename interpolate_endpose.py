import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['FreeSans']


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.fc1(x)
        y = self.sigmoid(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.fc3(y)
        return y


if __name__ == '__main__':
    x_train_numpy = np.load('data/x_train.npy').astype(np.float32)
    y_train_numpy = np.load('data/y_train.npy').astype(np.float32)
    x_train = torch.from_numpy(x_train_numpy)
    y_train = torch.from_numpy(y_train_numpy)

    model = Feedforward(3, 12, 20)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.eval()
    y_pred = model(x_train)
    before_train = criterion(y_pred.squeeze(), y_train)

    model.train()
    total_epoch = 100000
    show_progress = False
    loss_records = []
    for epoch in range(total_epoch):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred.squeeze(), y_train)
        loss.backward()
        if show_progress:
            print('Epoch {}: Loss: {}'.format(epoch, loss.item()))
        loss_records.append(loss.detach().numpy())
        optimizer.step()

    model.eval()
    y_pred = model(x_train)
    y_pred_numpy = y_pred.detach().numpy()
    after_train = criterion(y_pred.squeeze(), y_train)
    print('Loss Before:', before_train.item())
    print('Loss After:', after_train.item())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_records)
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Mean Squared Loss')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_train_numpy[:, 0], y_train_numpy[:, 0], 'x', label='True')
    ax.plot(x_train_numpy[:, 0], y_pred_numpy[:, 0], 'x', label='Predicted')
    ax.set_xlabel('X[0]')
    ax.set_ylabel('Y[0]')
    ax.legend()

    plt.show()
