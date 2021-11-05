import torch
import numpy as np
from sklearn.metrics import accuracy_score


class NeuralNetwork(torch.nn.Module):
    """
    """

    def __init__(self, idim, odim, dims=8):
        """
        Initialize the graph elements of the neural network
        we want to use to classify vectorized data.

        Parameters
        ----------
        idim : integer
            Number of input features. Must be positive.
        odim : integer
            Number of output features. Must be positive.
        """

        # Pytorch keeps track of the submodules (e.g. Linear)
        # you will write in your custom module. Under the hood,
        # the graph corresponding to your model is automatically built.
        # The nested modules will be added to an OrderedDict _modules
        # which is initialized in torch.nn.Module.__init__ function.
        # If torch.nn.Module.__init__ is not called (self._modules would
        # equal to None), when trying to add a Module, it will raise
        # an error (no key can be added to None).
        super(NeuralNetwork, self).__init__()

        assert isinstance(idim, int) and idim > 0
        assert isinstance(odim, int) and odim > 0

        if isinstance(dims, int):
            dims = [dims]

        layers = []
        last_dim = None
        num_layers = len(dims) + 1
        for i in range(num_layers):
            curr_layer: torch.nn.Linear = None
            if i == 0:
                d = dims[i - 1]
                curr_layer = torch.nn.Linear(idim, d)
                last_dim = d
                layers.append(curr_layer)
                layers.append(torch.nn.ReLU())
            elif i == num_layers - 1:
                curr_layer = torch.nn.Linear(last_dim, odim)
                layers.append(curr_layer)
            else:
                d = dims[i - 1]
                curr_layer = torch.nn.Linear(last_dim, d)
                last_dim = d
                layers.append(curr_layer)
                layers.append(torch.nn.ReLU())

        self.classifier = torch.nn.Sequential(*layers)

        # weights initialization
        def init_weights(module, mode="normal"):
            if mode == "normal":
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.normal_(0, 0.01)
                    module.bias.data.fill_(0)
                else:
                    pass
            else:
                raise NotImplementedError()

        self.apply(init_weights)

    def forward(self, x_data):
        """
        The super class torch.nn.Module has the forward function which
        needs to be implemented according to the custom graph we want
        to evaluate.
        At each time step, the graph is built allowing to dynamically
        change the behavior of the network.

        Parameters
        ----------
        x_data : torch.FloatTensor or torch.cuda.FloatTensor
            Batch data to analyze with shape BxF where B is the
            size of the batch and F the number of features.

        Example
        -------
        Generate 100 random tensors with 3 dimensions.
        B (batch_size) is 100 while F (features) is 3.

            B = 100
            F = 3
            xin = torch.FloatTensor(B,F)

            print(xin.shape)
            [Out] torch.Size([100, 3])

        Initialize a neural classifier to associate vectorized data
        to a set of 5 classes.
        C (number of classes) is 5.

            C = 5
            net = NeuralNetwork(F,C)

            yin = net(xin)

            print(yin.shape)
            [Out] torch.Size([100, 5])
        """

        y_scores = self.classifier(x_data)
        y_probs = torch.nn.Softmax(dim=1)(y_scores)

        return y_probs

    def evaluate(self, x_data, y_data, device=torch.device("cpu")):
        """
        Predict the labels of x_data and compute the accuracy
        using y_data as reference.

        Parameters
        ----------
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

        assert len(x_data.shape) == 2
        assert len(y_data.shape) == 1

        # store the training flag (from torch.nn.Module)
        mode = self.training
        self.eval()

        # convert to pytorch
        xin = torch.from_numpy(x_data).type(torch.FloatTensor).to(device)
        # forward pass - compute test predictions
        yout = self.forward(xin)
        # convert to numpy
        # IMPORTANTE: se non ci fosse .cpu() non riuscirebbe a convertire da CUDA a numpy, perché numpy funziona solo su CPU
        y_prob = yout.detach().cpu().numpy().squeeze()
        # get predicted labels
        y_pred = np.argmax(y_prob, axis=1)
        # compute accuracy
        accuracy = accuracy_score(y_data, y_pred)

        if mode:
            self.train()

        return y_pred, accuracy
