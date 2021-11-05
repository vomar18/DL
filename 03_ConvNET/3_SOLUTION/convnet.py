import torch
import numpy as np
from sklearn.metrics import accuracy_score


class CNN(torch.nn.Module):

    def __init__(self, inshape, nclasses):
        """
        Initialize the graph elements of the convolutional
        neural network we want to use to classify image data.

        Parameters
        ----------
        inshape : tuple
            Shape of the input image in the format CxHxW.
        nclasses : integer
            Number of output classes. Must be positive.
        """

        # Pytorch keeps track of the submodules (e.g.  Linear)
        # you will write in your custom module.  Under the hood,
        # the graph corresponding to your model is automatically built.
        # The nested modules will be added to an OrderedDict _modules
        # which is initialized in torch.nn.Module.__init__ function.
        # If torch.nn.Module.__init__ is not called (self._modules would
        # equal to None), when trying to add a Module, it will raise
        # an error (no key can be added to None).
        super(CNN, self).__init__()

        assert isinstance(inshape, tuple)
        assert len(inshape) == 3
        for i in range(3):
            assert isinstance(inshape[i], int)
        assert inshape[0] > 0 and inshape[1] > 0 and inshape[2] > 0
        assert isinstance(nclasses, int) and nclasses > 0

        ch = inshape[0] # number of input channels (depth of the convolutional filters)
        nkernels = 16 # number of convolutional filters
        ksize = 3 # filter size (square)
        
        # create convolutional layer with 'nkernels' filters of shape
        # 'ksize*ksize'
        self.conv1 = torch.nn.Conv2d(ch, nkernels, ksize)

        # Question 3) what is the number of features? 
        nfts = ...

        # create the linear layer (actual classifier)
        self.linear1 = torch.nn.Linear(nfts, nclasses)

        self.activation = torch.nn.ReLU()

        # weights initialization
        def init_weights(module, mode="normal"):
            if mode == "normal":
                if isinstance(module, torch.nn.Linear):
                    module.weight.data.normal_(0, 0.01)
                    module.bias.data.fill_(0)
                elif isinstance(module, torch.nn.Conv2d):
                    # xavier initialization for convolutional filters
                    torch.nn.init.xavier_uniform_(module.weight)
                    module.bias.data.fill_(0.)
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
            Batch data to analyze with shape BxCxHxW where B is the
            size of the batch, C the number of channels and
            HxW the image pixel resolution.

        Example
        -------
        Generate 100 random tensor images of size 640x480.
        B (batch_size) is 100 while H and W are 480 and 640.

            B = 100
            C, H, W = 3, 480, 640
            xin = torch.FloatTensor(B,C,H,W)

            print(xin.shape)
            [Out] torch.Size([100, 3, 640, 480])

        Initialize a neural classifier to associate vectorized data
        to a set of 5 classes.
        C (number of classes) is 5.

            C = 5
            net = CNN(F,C)

            yin = net(xin)

            print(yin.shape)
            [Out] torch.Size([100, 5])
        """

        y1 = self.conv1(x_data)
        y1 = self.activation(y1)
        y1 = y1.view(-1, np.prod(y1.shape[1:])) # vectorize the final feature map
        y_scores = self.linear1(y1)
        y_probs = torch.nn.Softmax(dim=1)(y_scores)

        return y_probs


    def evaluate(self, x_data, y_data, device=torch.device("cpu")):
        """
        Predict the labels of x_data and compute the accuracy
        using y_data as reference.

        Parameters
        ----------
        x_data : numpy.ndarray
            NxCxHxW matrix containind the testing data to evaluate.
        y_data : numpy.ndarray
            (N,) array containing the true labels of x_data.

        Returns
        -------
        y_pred : numpy.ndarray
            (N,) array containing the predicted labels.
        accuracy : float
            Classification accuracy.
        """

        assert len(x_data.shape) == 4 # batch - channels - height - width
        assert len(y_data.shape) == 1

        # store the training flag (from torch.nn.Module)
        mode = self.training
        self.eval()

        # convert to pytorch
        xin = torch.from_numpy(x_data).type(torch.FloatTensor).to(device)
        # forward pass - compute test predictions
        yout = self.forward(xin)
        # convert to numpy
        y_prob = yout.detach().cpu().numpy().squeeze()
        # get predicted labels
        y_pred = np.argmax(y_prob,axis=1)
        # compute accuracy
        accuracy = accuracy_score(y_data, y_pred)

        if mode:
            self.train()

        return y_pred, accuracy
