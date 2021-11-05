import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import DataGenerator
from network import NeuralNetwork


if __name__ == "__main__":

    # fix seed for random generator
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # generate noisy data
    dg = DataGenerator()
    dataset = dg.load("notmnist")
    trainset, testset = dataset
    # prepare train data
    x_train = trainset[:,:-1]
    y_train = trainset[:,-1].reshape(-1)
    # prepare test data
    x_test = testset[:,:-1]
    y_test = testset[:,-1].reshape(-1)

    # regressor initialization
    nfeatures = x_train.shape[1]
    nclasses = len(set(y_train))
    model = NeuralNetwork(nfeatures, nclasses, dims=[128, 64])

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    #================================================================
    plt.figure()
    plt.suptitle("notMNIST samples")
    idx = np.random.permutation(x_train.shape[0])[:9]
    for i,xi in enumerate(idx):
        plt.subplot(331 + i)
        size = np.sqrt(nfeatures).astype(int)
        image = x_train[xi].reshape(size,size)
        plt.imshow(image)
    plt.show()
    #================================================================

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # create loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    #================================================================
    # add variable to store history
    history = {}
    history["accuracy"] = []
    history["loss"] = []
    #================================================================

    # train the linear regressor on trainset
    model.train()
    epochs = 1000
    for ep in range(epochs):
        # convert to pytorch
        xin = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
        yin = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
        # forward pass - compute train predictions
        yout = model(xin)
        # compute loss value
        loss = criterion(yout, yin)
        # backward pass - update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #============================================================
        if ep % 50 == 0:
            y_pred, train_acc = model.evaluate(x_train, y_train, device)
            history["accuracy"].append(train_acc)
            history["loss"].append(loss.detach().cpu().numpy().item())
            xx = np.arange(len(history["accuracy"]))
            # clear figure
            plt.clf()
            title = "Epoch " + str(ep) + "\nTrain accuracy: {:.2f}".format(train_acc)
            plt.suptitle(title)
            # show accuracy
            plt.subplot(121)
            plt.grid(True)
            plt.xlabel("Accuracy")
            plt.plot(xx, history["accuracy"])
            # show loss
            plt.subplot(122)
            plt.grid(True)
            plt.xlabel("Loss")
            plt.plot(xx, history["loss"])
            # update and display figure
            plt.pause(0.2)
        #============================================================

    # compute accuracy score on train and test sets
    _, train_acc = model.evaluate(x_train, y_train, device)
    y_pred, test_acc = model.evaluate(x_test, y_test, device)

    # visualize final results
    print("Train accuracy: {:.2f}".format(train_acc) + " -- Test accuracy: {:.2f}".format(test_acc))
    plt.show()