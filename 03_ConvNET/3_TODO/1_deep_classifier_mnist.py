
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import DataGenerator
from network import NeuralNetwork

"""
Follow the script and fulfill the TODOs and NotImeplementedError

Here we will see a deep classifier for images (MNIST dataset, 
see https://en.wikipedia.org/wiki/MNIST_database ) using a fully
connected neural network.

"""

# this is fundamental for execute the debugger!
if __name__ == "__main__":

    # fix seed for random generator
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # generate noisy data
    dg = DataGenerator() # carica la classe particolare che contiene funzioni utili
    dataset = dg.load("notmnist") # caricati il dataset notmnist (dizionario formato da label and data)
    trainset, testset = dataset # that is simple data and target
    # prepare train data
    x_train = trainset[:, :-1] # tutte le righe, tutte le colonne tranne l'ultima
    y_train = trainset[:, -1].reshape(-1) # crea un vettore riga semplice tutte le righe, utlima colonna
    # prepare test data
    x_test = testset[:, :-1] # tutte le righe, tutte le colonne tranne l'ultima
    y_test = testset[:, -1].reshape(-1) # crea un vettore riga semplice tutte le righe, utlima colonna

    # regressor initialization
    nfeatures = x_train.shape[1]
    nclasses = len(set(y_train)) # con set conto quanti ele sono == e li unisco
    model = NeuralNetwork(nfeatures, nclasses) # attivo NN from file network.py

    device = torch.device("cpu") # use gpu instead of cpu --> but this network is very simple!
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # ================================================================
    plt.figure()
    plt.suptitle("notMNIST samples")
    idx = np.random.permutation(x_train.shape[0])[:9] # prendi i primi 9 valori casuali
    for i, xi in enumerate(idx):
        plt.subplot(331 + i)  # crei un grafico con 3x3x1 grafici
        size = np.sqrt(nfeatures).astype(int) # dimensione ogni singola immagine
        image = x_train[xi].reshape(size, size)
        plt.imshow(image)
    plt.show()
    # ================================================================

    # create optimizer --> SGD is the stocastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # create loss criterion. 
    # Q.1  Which task is this? Which loss do we need?
    # TODO: see https://pytorch.org/docs/stable/nn.html#loss-functions
    # what type of formula fot the loss ypu want to use?
    criterion = torch.nn.CrossEntropyLoss()
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
        # torch wants everything into a torch datatype!! so
        # convert to pytorch
        xin = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
        yin = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
        # forward pass - compute train predictions
        yout = model(xin) # is the linearizzation of the image!
        # compute loss value
        loss = criterion(yout, yin)
        # backward pass - update the weights
        optimizer.zero_grad() # set gradient coeff.s to none instead of zero
        loss.backward() # calcola tutte le varie derivate parziali collegate all'errore
        optimizer.step() # esegue l'aggiornamento di tutti i pesi come fatto in aula
        # after that you have the new model

        # ============================================================
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
        # ============================================================

    # clear the tree of partial derivates and save memory and start to evaluate mode
    # important
    # compute accuracy score on train and test sets
    _, train_acc = model.evaluate(x_train, y_train, device)
    y_pred, test_acc = model.evaluate(x_test, y_test, device)

    # Q.2  Why the accuracy is so low? How can we increase the performances?
    # TODO: Modify the architecture, keeping the epochs at 1000 to increase the performance.
    # Try changing activation functions, or dimensions of Linear() or even the number of linear

    # visualize final results
    print(
        "Train accuracy: {:.2f}".format(train_acc)
        + " -- Test accuracy: {:.2f}".format(test_acc)
    )
    plt.show()
