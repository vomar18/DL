def LSRegression(X, y):
    # ------------------ Least Squares Estimation ------------------
    # we have seen the least square solution to Y = XA + E
    # (X^T*X)*theta - X^T*Y = 0 --> W == theta

    ones = np.ones((y.size, 1)) # colonna di 1 [506 x 1]
    X_aug = np.block([X, ones]) # X_aug diventa [ X | colonna di 1 ] incremento dimensione di X
    print(X_aug.shape) # [506 x 3]
    W = np.linalg.inv(X_aug.T.dot(X_aug)).dot(X_aug.T).dot(y)
    print("Projection matrix:", W, " ultimo è bias\n")

    return W


def perform_lr_ls(X, Y):
    """
    Given `X` - matrix of shape (N,D) of input features
          `Y` - target y values
    Solves for linear regression using the Least Squares algorithm. implemented in LSRegression
    Returns weights and prediction.
    """
    W = LSRegression(X, Y)
    # Y_prediction casa prezzo di 506 case
    Y_pred = np.dot(W[0:-1],X.T) + W[-1]  # W[-1] è bias è l'ultimo elemento del vettore W

    # puoi calcolarti quant'è la perdita che hai nella tua predizione
    loss_ls = mean_squared_error(Y, Y_pred)
    print("performance of this LS:")
    print("MSE is:", format(loss_ls))

    plt.figure(figsize=(10, 10))
    plt.scatter(Y, Y_pred)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("LS solution")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    #plot3d_lr(W, X, "LS LR")
    return W, Y_pred
