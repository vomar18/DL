import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from utils import DataGenerator

from convnet_solutions import CustomCNN

BATCH = True
BATCH_SIZE = 150
EPOCHS = 300

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

    # extract data shape
    nsamples = x_train.shape[0]
    nfeatures = x_train.shape[1]
    nclasses = len(set(y_train))

    # reshape data to image format
    size = int(np.sqrt(nfeatures))
    x_train = x_train.reshape(x_train.shape[0], 1, size, size)
    x_test = x_test.reshape(x_test.shape[0], 1, size, size)

    # Q.  what is the value "1" in the reshape function?
    # Ottengo un input 1xHxW, quindi è per avere l'input corretto

    # znorm
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    x_train = x_train - mu
    x_train = x_train / var
    x_test = x_test - mu
    x_test = x_test / var

    # subsampling
    nsamples = int(nsamples*0.1)
    x_train = x_train[:nsamples]
    y_train = y_train[:nsamples]

    # initialize model
    inshape = (1,size,size)
    model = CustomCNN(inshape, nclasses)

    # [optional] set destination device (cpu vs gpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    #================================================================
    # add variable to store history
    history = {}
    history["accuracy"] = []
    history["loss"] = []
    #================================================================

    tensor_x = torch.from_numpy(x_train)
    tensor_y = torch.from_numpy(y_train)
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True) # create your dataloader

    for ep in range(EPOCHS):
        model.train()
        for sample, target in tqdm(my_dataloader):
            # convert to pytorch
            xin = sample.type(torch.FloatTensor).to(device)
            yin = target.type(torch.LongTensor).to(device)
            # forward pass - compute train predictions
            yout = model(xin)
            # compute loss value
            loss = criterion(yout, yin)
            # backward pass - update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #============================================================
        if ep % 5 == 0:
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
