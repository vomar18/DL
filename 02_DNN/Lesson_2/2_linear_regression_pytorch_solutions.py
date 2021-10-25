import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import DataGenerator


if __name__ == "__main__":

    # fix seed for random generator
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # paramset initialization
    npoints = 100
    slope = 2 * np.random.rand() - 1
    offset = 2 * np.random.rand() - 1

    # generate noisy data
    dg = DataGenerator()
    dataset = dg.generate_linear_data(npoints, slope, offset)
    trainset, testset = dataset
    # prepare train data
    x_train = trainset[:,0].reshape(-1,1)
    y_train = trainset[:,1].reshape(-1,1)
    # prepare test data
    x_test = testset[:,0].reshape(-1,1)
    y_test = testset[:,1].reshape(-1,1)

    #================================================================
    plt.figure()
    plt.grid(True)
    plt.title("Dataset")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x_train, y_train, c="b", label="train")
    plt.scatter(x_test, y_test, c="g", label="test")
    plt.legend()
    plt.show()
    #================================================================

    # regressor initialization
    model = torch.nn.Linear(1,1)

    # weights initialization
    model.weight.data.normal_(0, 0.01)
    model.bias.data.fill_(0)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    #================================================================
    # add variable to store history
    history = {}
    history["weight"] = []
    history["bias"] = []
    #================================================================

    # train the linear regressor on trainset
    model.train()
    epochs = 1000
    for ep in range(epochs):
        # convert to pytorch
        xin = torch.from_numpy(x_train)
        yin = torch.from_numpy(y_train)
        # forward pass - compute train predictions
        yout = model(xin)
        # compute loss value
        loss = torch.mean(torch.pow(yout - yin, 2))
        # backward pass - update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #============================================================
        if ep % 50 == 0:
            # convert to numpy
            y_pred = yout.detach().numpy()
            # clear figure
            plt.clf()
            plt.suptitle("Epoch " + str(ep))
            plt.tight_layout()
            # visualize training model optimization
            plt.subplot(121)
            plt.grid(True)
            plt.title("Training")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.scatter(x_train, y_train, c="b", label="train")
            plt.plot(x_train, y_pred, c="r", label="model")
            plt.legend()
            # visualize weights (and biases)
            plt.subplot(122)
            plt.grid(True)
            plt.title("Weights")
            plt.xlabel("slope")
            plt.ylabel("offset")
            model_dict = dict(model.state_dict())
            w = model_dict["weight"].squeeze().item()
            b = model_dict["bias"].squeeze().item()
            history["weight"].append(w)
            history["bias"].append(b)
            plt.plot(history["weight"], history["bias"], c="b")
            plt.scatter(history["weight"], history["bias"], c="b")
            # update and display figure
            plt.pause(0.2)
        #============================================================

    # compute the predictions on testset
    model.eval()
    # convert to pytorch
    xin = torch.from_numpy(x_test)
    # forward pass - compute test predictions
    yout = model(xin)
    # convert to numpy
    y_pred = yout.detach().numpy()

    # visualize final results
    plt.figure()
    plt.grid(True)
    plt.title("Least Squares Linear regression via PYTORCH")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x_train, y_train, c="b", label="train")
    plt.scatter(x_test, y_test, c="g", label="test")
    plt.plot(x_test, y_pred, c="r", label="model")
    plt.legend()
    plt.show()
