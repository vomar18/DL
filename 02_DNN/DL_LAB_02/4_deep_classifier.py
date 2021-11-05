import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import DataGenerator
from network import NeuralNetwork


"""
Follow the script and fulfill the TODOs and NotImeplementedError

In this script we will see a deep classifier (fully connected neural network)
which performs multi-class classification at the same way as we saw for
the multinomial logistic classifier
"""

if __name__ == "__main__":

    # fix seed for random generator
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # paramset initialization
    nfeatures = 2
    nclasses = 3
    npoints = 200
    spread_factor = 12

    # generate noisy data
    dg = DataGenerator()
    dataset = dg.generate_gaussian_data(nclasses, npoints, spread_factor)
    trainset, testset = dataset
    # prepare train data
    x_train = trainset[:, :nfeatures].reshape(-1, nfeatures)
    y_train = trainset[:, -1].reshape(-1)
    # prepare test data
    x_test = testset[:, :nfeatures].reshape(-1, nfeatures)
    y_test = testset[:, -1].reshape(-1)

    # regressor initialization
    model = NeuralNetwork(nfeatures, nclasses)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # create loss criterion for multi-class classification: which loss should we use?
    # TODO: see https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion = None
    assert criterion is not None

    # ================================================================
    # add variable to store history
    history = {}
    history["accuracy"] = []
    history["loss"] = []
    # ================================================================

    # train the linear regressor on trainset
    model.train()
    epochs = 1000
    for ep in range(epochs):
        # convert to pytorch
        xin = torch.from_numpy(x_train).type(torch.FloatTensor)
        yin = torch.from_numpy(y_train).type(torch.LongTensor)
        # forward pass - compute train predictions
        yout = model(xin)
        # compute loss value
        loss = criterion(yout, yin)
        # backward pass - update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ============================================================
        if ep % 50 == 0:
            y_pred, train_acc = model.evaluate(x_train, y_train)
            history["accuracy"].append(train_acc)
            history["loss"].append(loss.detach().numpy().item())
            xx = np.arange(len(history["accuracy"]))
            # clear figure
            plt.clf()
            title = "Epoch " + str(ep) + "\nTrain accuracy: {:.2f}".format(train_acc)
            plt.suptitle(title)
            # visualize data samples
            plt.subplot(221)
            plt.grid(True)
            plt.ylabel("Dataset")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
            # visualize train predictions
            plt.subplot(222)
            plt.grid(True)
            plt.ylabel("Predictions")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_pred)
            # show accuracy
            plt.subplot(223)
            plt.grid(True)
            plt.ylabel("Accuracy")
            plt.plot(xx, history["accuracy"])
            # show loss
            plt.subplot(224)
            plt.grid(True)
            plt.ylabel("Loss")
            plt.plot(xx, history["loss"])
            # update and display figure
            plt.pause(0.2)
        # ============================================================

    # compute accuracy score on train and test sets
    _, train_acc = model.evaluate(x_train, y_train)
    y_pred, test_acc = model.evaluate(x_test, y_test)

    # visualize final results
    plt.figure()
    plt.tight_layout()
    title = "Deep Classification via PYTORCH"
    title = (
        title
        + "\nTrain accuracy: {:.2f}".format(train_acc)
        + " -- Test accuracy: {:.2f}".format(test_acc)
    )
    plt.suptitle(title)
    # visualize data samples
    plt.subplot(121)
    plt.grid(True)
    plt.xlabel("Dataset")
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, alpha=0.2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c="r", s=80)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    # visualize data samples
    plt.subplot(122)
    plt.grid(True)
    plt.xlabel("Predictions")
    plt.scatter(x_train[:, 0], x_train[:, 1], c="b", alpha=0.2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c="r", s=80)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
    plt.show()
