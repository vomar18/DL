def perform_lr_sklearn(X, Y):
    lr_sk = LinearRegression()
    lr_sk.fit(X, Y)
    W = np.hstack((lr_sk.coef_, lr_sk.intercept_))
    Y_pred = lr_sk.predict(X)
    loss_sk = mean_squared_error(Y, Y_pred)
    print("Model performance SKLEARN LR:")
    print("MSE is:",format(loss_sk))

    plt.figure(figsize=(4, 3))
    plt.scatter(Y, Y_pred)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("SKLearn Linear")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    #plot3d_lr(W, X, "SKLearn")

    return W, Y_pred
