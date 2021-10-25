def gradfn(theta, X, Y):
    """
    Given `theta` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Return gradient of each weight evaluated at the current value
    """
    N, D = np.shape(X)
    Y_pred = np.matmul(X, theta)
    error = Y_pred - Y # ATT DEVE necessariamente essere così!
    #print("errore:", np.mean(error))
    return np.matmul(X.T,error) / float(N)


def MyGDregression(X, Y, niter, alpha):
    """
    Given `X` - matrix of shape (N,D) of input features --> NON HAI SOLO X PERCHÈ HAI BIAS! + DIMENSIONI!
          `y` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    """
    ones = np.ones((Y.size, 1))
    X_aug = np.block([X,ones]) # devi tenere conto dei bias!!
    N, D = np.shape(X_aug)
    # initialize all the weights to zeros
    theta = np.zeros([D]) # 1xD
    print("theta:", theta)
    print("finding coefficients: ...")
    for k in range(niter):
        d_theta = gradfn(theta, X_aug, Y)
        theta = theta - alpha * d_theta # alpha è il LEARNING RATE

    print("final theta:", theta)
    return theta