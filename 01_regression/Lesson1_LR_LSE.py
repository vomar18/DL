"""
# Exercise 0) 
#  - Run the script and load the boston dataset
#  - Look the EDA on the boston dataset
#
# Exercise 1)
#  - Implement the `LSRegression` function
#  - Use its output (W) to compute the projection for the input data X
#  - See visualization of values and hyperplane
#
# Exercise 2)
#  - Implement the `MyGDregression` function and `gradfn` as we have seen during the class.
#  - Use its output (W) to compute the projection for the input data X
#  - See visualization of values and hyperplane
#
# Exercise 3)
#  - Proof that Gradient Descent is faster w.r.t. Least Squares (hint: watch the sklearn function `make_regression`)
#

"""
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Global parameters
visualize = True
features = ["LSTAT", "RM"]
target_name = "MEDV"


def inspect_boston_dataset(
    visualize: bool = True,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # --------- LOADING AND FIRST UNDERSTANDING ---------
    # We can use the boston dataset already built in scikit-learn
    # Let's load it first

    boston = load_boston()
    # what is the boston ?: you can take a look by watching the variables in the debugger window,
    #  or use some functions

    print(boston.keys())
    # data: contains the information for various houses
    # target: prices of the house
    # feature_names: names of the features
    # DESCR: describes the dataset
    # filename: where is the original (csv) file

    print(boston.data.shape)
    # for example, this above should be omitted, and kept just in the pycharm debugger scope
    # this is for keep the code clean and essential

    print(boston.feature_names)

    # We can now access different information of the dataset like the data itself, the features and general information on
    # the dataset. Let's print out some general information on the dataset
    print(boston.DESCR)

    boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
    # pandas gives me crucial functions to perform data profiling
    print(boston_pd.head())
    # We can see that the target value MEDV is missing from the data. We create a new column of target values and add it to the dataframe.
    boston_pd[target_name] = boston.target
    # After loading the data, it’s a good practice to see if there are any missing values
    # in the data. We count the number of missing values for each feature using isnull()
    print(boston_pd.isnull().sum())

    # --------- DATA PROFILING: A DEEPER UNDERSTANDING ---------
    # Exploratory Data Analysis, or data profiling, is a very important step before training
    # the model. In this section, we will use some visualizations to understand the relationship
    # of the target variable with other features.

    # Let’s first plot the distribution of the target variable MEDV.
    # We will use the distplot function from the seaborn library.
    if visualize:
        sns.set(rc={"figure.figsize": (11.7, 8.27)})
        # sns.distplot(boston_pd[target_name], bins=30)
        sns.histplot(boston_pd[target_name], kde=True, stat="density", linewidth=0)
        plt.show(block=True)
        plt.close()
    # time.sleep(1)

    # We see that the values of MEDV are distributed normally with few outliers.

    # Without any preliminary analysis, we can access all the data by putting them in a numpy array.
    # (check homogeneity) This way we can create a matrix where
    # each row contains all 13 features for that entry and each column contains all the values a feature.
    # Of course, we need also the target (dependent variable) which we wish to model using our features
    X = np.array(boston.data, dtype="f")
    Y = np.array(boston.target, dtype="f")

    # As seen in class, there are some requirements that we need to fulfill in order to make use of linear regression.
    # The first one is that the data should display some form a linear relation. We can check this by performing a scatter
    # plot of each feature (x) and the labels (y). This is commonly referred to as a scatter matrix.
    if visualize:
        fig, axs = plt.subplots(7, 2, figsize=(14, 30))
        for index, feature in enumerate(boston.feature_names):
            subplot_idx = int(index / 2)
            if index % 2 == 0:
                axs[subplot_idx, 0].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 0].set_xlabel(feature)
                axs[subplot_idx, 0].set_ylabel("Target")
            else:
                axs[subplot_idx, 1].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 1].set_xlabel(feature)
                axs[subplot_idx, 1].set_ylabel("Target")
        plt.savefig("linearity_scatter_plots.png")
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    # Next we need to check if the data are co-linear. In linear regression high co-linearity between the features is a
    # problem. We can see how much the data is correlated by looking a correlation coefficient. Since our features are all
    # numerical, we'll use the famous Pearson correlation coefficient.

    # Next, we create a correlation matrix that measures the linear relationships between
    # the variables. The correlation matrix can be formed by using the corr function from
    # the pandas dataframe library. We will use the heatmap function from the seaborn library
    # to plot the correlation matrix.

    if visualize:
        correlation_matrix = boston_pd.corr().round(2)
        # annot = True to print the values inside the square
        sns.heatmap(data=correlation_matrix, annot=True)
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    if visualize:
        target = boston_pd[target_name]
        plt.figure(figsize=(20, 5))
        for i, col in enumerate(features):
            plt.subplot(1, len(features), i + 1)
            x = boston_pd[col]
            y = target
            plt.scatter(x, y, marker="o")
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel(target_name)
        # plt.savefig('sel_features_analysis.png')
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    X = boston_pd[features]
    Y = boston_pd[target_name]

    return boston, boston_pd, X, Y


def plot3d_lr(W, X, title):
    # create x,y,z data for 3d plot
    y_pred = np.sum(W[0:-1] * X, axis=1) + W[-1]
    if isinstance(X, pd.DataFrame):
        x = X[features[0]]
        y = X[features[1]]
    else:
        x = X[:, 0]  # LSTAT
        y = X[:, 1]  # RM
    z = y_pred
    data = np.c_[x, y, z]

    # Create X, Y data to predict
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    XX, YY = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))

    # calculate prediction
    Z = W[0] * XX + W[1] * YY + W[-1]
    # plot the surface
    fig = plt.figure()
    # ax = fig.gca(projection="3d")
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="r", s=50)
    plt.title(title)
    plt.xlabel(features[0])  # LSTAT
    plt.ylabel(features[1])  # RM
    ax.set_zlabel(target_name)
    plt.show()
    # plt.savefig('3d_plane_of_best_fit.png')
    plt.close()


def LSRegression(X, y):
    # ------------------ Least Squares Estimation ------------------
    W = None
    raise NotImplementedError()
    return W


def perform_lr_ls(X, Y):
    """
    Given `X` - matrix of shape (N,D) of input features
          `Y` - target y values
    Solves for linear regression using the Least Squares algorithm. implemented in LSRegression
    Returns weights and prediction.
    """
    W = LSRegression(X, Y)
    Y_pred = None

    raise NotImplementedError()

    plt.figure(figsize=(4, 3))
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

    plot3d_lr(W, X, "LS LR")
    return W, Y_pred


# SKLEARN Linear Regression model
def perform_lr_sklearn(X, Y):
    lr_sk = LinearRegression()
    lr_sk.fit(X, Y)
    W = np.hstack((lr_sk.coef_, lr_sk.intercept_))
    Y_pred = lr_sk.predict(X)
    loss_sk = mean_squared_error(Y, Y_pred)
    print("Model performance SKLEARN LR:")
    print("--------------------------------------")
    print("MSE is {}".format(loss_sk))
    print("\n")

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

    plot3d_lr(W, X, "SKLearn")

    return W, Y_pred


# Gradient Descent


def gradfn(theta, X, Y):
    """
    Given `theta` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Return gradient of each weight evaluated at the current value
    """
    raise NotImplementedError()


def MyGDregression(X, Y, niter, alpha):
    """
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    """
    raise NotImplementedError


def perform_lr_gd(X, Y, iters: int = 10000, alpha: float = 0.005):

    theta = MyGDregression(X, Y, iters, alpha)
    Y_pred_GD = None

    raise NotImplementedError()

    loss_sgd = mean_squared_error(Y, Y_pred_GD)
    print("Model performance GD:")
    print("--------------------------------------")
    print("MSE is {}".format(loss_sgd))
    print("\n")

    plt.figure(figsize=(4, 3))
    plt.scatter(Y, Y_pred_GD)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("GD")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    plot3d_lr(theta, X, "GD")

    return W, Y_pred


def residual_analisys(Y, Y_pred):

    # Now that we have the results, we can also analyze the residuals
    residuals = Y - Y_pred

    # First thing we can easily do is check the residuals distribution.
    # We expect to see a normal or "mostly normal" distribution.
    # For this we can use a histogram
    # ax = sns.distplot(residuals, kde=True)
    ax = sns.histplot(residuals, kde=True, stat="density", linewidth=0)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    plt.title("Residuals distribution")
    # plt.savefig("regression_residual_kde.png")
    plt.show()
    plt.close()

    # Lastly, we can check if the residuals behave correctly with a normal probability plot
    # This can be very useful when we have few samples, as this graph is more sensitive than the histogram.
    # We can easily calculate the theoretical values needed for the normal probability plot.
    fig = plt.figure()
    res = stats.probplot(residuals, plot=plt, fit=False)
    # plt.savefig("residual_normality_plot.png")
    plt.show()
    plt.close()


boston_raw, boston_pd, X, Y = inspect_boston_dataset(visualize)


# Data non-normalized
W, Y_pred = perform_lr_ls(X, Y)
# residual_analisys(Y, Y_pred)
print("R2 score", r2_score(Y, Y_pred))

W, Y_pred = perform_lr_sklearn(X, Y)
# residual_analisys(Y, Y_pred)
print("R2 score", r2_score(Y, Y_pred))

T, Y_pred = perform_lr_gd(X, Y)
# residual_analisys(Y, Y_pred)
print("R2 score", r2_score(Y, Y_pred))

# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Data normalized
W_norm, Y_pred_norm = perform_lr_ls(X, Y)
# residual_analisys(Y, Y_pred_norm)
print("R2 score", r2_score(Y, Y_pred_norm))

W_norm, Y_pred_norm = perform_lr_sklearn(X, Y)
# residual_analisys(Y, Y_pred_norm)
print("R2 score", r2_score(Y, Y_pred_norm))

T_norm, Y_pred_norm = perform_lr_gd(X, Y)
# residual_analisys(Y, Y_pred_norm)
print("R2 score", r2_score(Y, Y_pred_norm))

print(T, T_norm)

# ----------------------- LS vs GD -----------------------

# TODO
