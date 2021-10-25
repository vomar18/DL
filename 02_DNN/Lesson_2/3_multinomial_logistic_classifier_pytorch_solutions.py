import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils import DataGenerator


def evaluate(classifier, x_data, y_data):
    """
    Predict the labels of x_data and compute the accuracy
    using y_data as reference.

    Parameters
    ----------
    classifier : torch.nn.Module
        PyTorch module to evaluate.
    x_data : numpy.ndarray
        NxF matrix containind the testing data to evaluate.
    y_data : numpy.ndarray
        (N,) array containing the true labels of x_data.

    Returns
    -------
    y_pred : numpy.ndarray
        (N,) array containing the predicted labels.
    accuracy : float
        Classification accuracy.
    """

    assert isinstance(classifier, torch.nn.Module)
    assert len(x_data.shape) == 2
    assert len(y_data.shape) == 1

    # compute the predictions on testset
    classifier.eval()
    # convert to pytorch
    xin = torch.from_numpy(x_data).type(torch.FloatTensor)
    # forward pass - compute test predictions
    yout = classifier(xin)
    # convert to numpy
    y_prob = yout.detach().numpy().squeeze()
    # get predicted labels
    y_pred = np.argmax(y_prob,axis=1)
    # compute accuracy
    accuracy = accuracy_score(y_data, y_pred)
    return y_pred, accuracy


if __name__ == "__main__":

    # fix seed for random generator
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # paramset initialization
    nclasses = 3
    npoints = 200
    spread_factor = 12

    # generate noisy data
    dg = DataGenerator()
    dataset = dg.generate_gaussian_data(nclasses, npoints, spread_factor)
    trainset, testset = dataset
    # prepare train data
    x_train = trainset[:,:2].reshape(-1,2)
    y_train = trainset[:,2].reshape(-1)
    # prepare test data
    x_test = testset[:,:2].reshape(-1,2)
    y_test = testset[:,2].reshape(-1)

    # regressor initialization
    nfeatures = 2
    model = torch.nn.Linear(nfeatures, nclasses)

    # weights initialization
    model.weight.data.normal_(0, 0.01)
    model.bias.data.fill_(0)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # create loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    # train the linear regressor on trainset
    model.train()
    epochs = 1000
    for ep in range(epochs):
        # convert to pytorch
        xin = torch.from_numpy(x_train).type(torch.FloatTensor)
        yin = torch.from_numpy(y_train).type(torch.LongTensor)
        # forward pass - compute train predictions
        yout = model(xin)

        # In yout we have the features, but we need a proability distribution.
        # Therefore, if you remember the lesson, we need to apply a function called Softmax 
        yout = torch.nn.Softmax(dim=1)(yout)

        # compute loss value
        loss = criterion(yout, yin)

        # backward pass - update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            y_pred, test_acc = evaluate(model, x_test, y_test)
            plt.clf()
            plt.grid(True)
            plt.xlabel("Predictions")
            # plt.scatter(x_train[:,0], x_train[:,1], c="b", alpha=0.2)
            # plt.scatter(x_test[:,0], x_test[:,1], c="r", s=80)
            plt.scatter(x_test[:,0], x_test[:,1], c=y_pred)
            plt.pause(0.2)

    # compute accuracy score on train and test sets
    _, train_acc = evaluate(model, x_train, y_train)
    y_pred, test_acc = evaluate(model, x_test, y_test)

    # visualize final results
    plt.close()
    plt.figure()
    plt.tight_layout()
    title = "Multinomial Logistic Classification via PYTORCH"
    title = title + "\nTrain accuracy: {:.2f}".format(train_acc) + " -- Test accuracy: {:.2f}".format(test_acc)
    plt.suptitle(title)
    # visualize data samples
    plt.subplot(121)
    plt.grid(True)
    plt.xlabel("Dataset")
    plt.scatter(x_train[:,0], x_train[:,1], c=y_train, alpha=0.2)
    plt.scatter(x_test[:,0], x_test[:,1], c="r", s=80)
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test)
    # visualize data samples
    plt.subplot(122)
    plt.grid(True)
    plt.xlabel("Predictions")
    plt.scatter(x_train[:,0], x_train[:,1], c="b", alpha=0.2)
    plt.scatter(x_test[:,0], x_test[:,1], c="r", s=80)
    plt.scatter(x_test[:,0], x_test[:,1], c=y_pred)
    plt.show()