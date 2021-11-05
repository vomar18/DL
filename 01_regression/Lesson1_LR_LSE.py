"""
# Exercise 0) 
#  - Run the script and load the boston dataset
#  - Look the EDA on the boston dataset
#
# Exercise 1)
#  - Implement the `LSRegression` function
#  - Use its output (W==theta) to compute the projection for the input data X
#    the projection is like: tilde(Y) = (W^T)X + theta_0
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

from sklearn.datasets import load_boston # ti serve un dataset
import pandas as pd # utilizzi la funzione pd.DataFrame per creare una tabella organizzata dei dati del dataset
import seaborn as sns # seaborn: statistical data visualization --> ti crea istogrammi (vedi loro risultato)
import matplotlib.pyplot as plt # crea un'immagine con più grafici(scatter) per vedere dopo puoi applicare linear regression
# APPLICHI LINEAR REGRESSION QUANDO:
# 1. hai un grafico lineare
# 2. i dati sono homoskedastic --> varianza fissa e limitata --> hai un limite nella distribuzione dei dati
# 3. gli errori di scarto (residual) sono indipendenti tra loro (questo caso non analizzato!)
# 4. l'errore residuo è distribuito come una pdf normale (questo caso non analizzato!)
# NON APPLICHI LINEAR REGRESSION:
# 1. l'andamento dei dati è non lineare
# 2. l'errore dei residui tende a + inf --> non limitato
# 3. dati non continui hai un distacco
import numpy as np # utilizzi matmul (matrix mult vera) e block ad esempio (espandi a dx vettore)
# confronto con libreria sklearn
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler # utile per sclare i dati così tutte features hanno peso uguale

import time # utile per capire dopo durata di velocità nel calcolo
from typing import Any, Dict, Tuple
from scipy import stats # utile per analisi dei residual con metoro stats.probplot

# Global parameters
visualize = True # only for visualize the graphs
features = ["LSTAT", "RM"] # try to predict the cost of the house just by using only the LSTAT and RM
target_name = "MEDV" # this is the median value that is a dependent value E' QUELLO CHE DEVI RICREARE


def inspect_boston_dataset( visualize: bool = True) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --------- LOADING AND FIRST UNDERSTANDING ---------
    # We can use the boston dataset already built in scikit-learn Let's load it first
#--> 1. ANALIZZARE IL DATASET SUO CONTENUTO ETC
    boston = load_boston()      # carichi il dataset
    print("dataset contiene:",boston.keys())            # tipologia dei dati che possiedi
    # data: contains the information for various houses
    # target: prices of the house
    # feature_names: names of the features
    # DESCR: describes the dataset
    # filename: where is the original (csv) file
    print("dimensioni dei dati:",boston.data.shape)     # dimensioni dei dati che possiedi
    print("tipologia di dati:",boston.feature_names)    # nomenclatura dei dati
    print("dataset description:",boston.DESCR)          # description
    # best to convert all data into a table made by []
    boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
    print(boston_pd.head()) # visualizza solo i primi 10 dati per ogni tipologia di dato
    # !! We can see that the target value MEDV is missing from the data.
    # !! We create a new column of target values and add it to the dataframe.
    boston_pd["MEDV"] = boston.target

#--> 2. VISUALIZZARE EVENTUALI DATI NULLI --> non esegue altre operazioni su dati nulli?
    # After loading the data, it’s a good practice to see if there are any missing values
    # in the data. We count the number of missing values for each feature using isnull()
    print("total null data per categoria:\n",boston_pd.isnull().sum())

    # --------- DATA PROFILING: A DEEPER UNDERSTANDING ---------
    # Exploratory Data Analysis, or data profiling, is a very important step before training
    # the model. In this section, we will use some visualizations to understand the relationship
    # of the target variable with other features.

 #--> 3. ANALIZZARE IL VALORE TARGET DEL DATASET # Let’s first plot the distribution of the target variable MEDV.
    if visualize:
        sns.histplot(boston_pd["MEDV"], kde=True, stat="density", linewidth=0) # seaborn: statistical data visualization
        plt.title("Density of Median value of owner-occupied homes in $1000's")
        plt.xlabel("costo delle case in 1000$")
        plt.ylabel("quantità di proprietari (in centinaia??)")
        plt.show(block=True)
        plt.close()
    # We see that the values of MEDV are distributed normally with few outliers.

#--> 4. CONVERTI TUTTI I DATI IN ARRAY DI NUMPY PER OTTIMIZZARE IL CALCOLO/ANALISI
    # Without any preliminary analysis, we can access all the data by putting them in a numpy array.
    # (check homogeneity) This way we can create a matrix where
    # each row contains all 13 features for that entry and each column contains all the values a feature.
    # Of course, we need also the target (dependent variable) which we wish to model using our features
    X = np.array(boston.data, dtype="f")
    Y = np.array(boston.target, dtype="f")

# --> 5. VISUALIZZA TUTTI I DATI PRESENTI NEL DATASET
    # As seen in class, there are some requirements that we need to fulfill in order to make use of linear regression.
    # The first one is that the data should display some form a linear relation. We can check this by performing a scatter
    # plot of each feature (x) and the labels (y). This is commonly referred to as a scatter matrix.
    if visualize:
        fig, axs = plt.subplots(7, 2, figsize=(14, 30)) # [RIGA, COLONNA]
        for index, feature in enumerate(boston.feature_names): # enumerates rende contabile la lista di nomi
            subplot_idx = int(index / 2)
            if index % 2 == 0:
                axs[subplot_idx, 0].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 0].set_xlabel(feature) # ricordati quando utilizzi axs devi usare set_xlabel !!
                axs[subplot_idx, 0].set_ylabel("Target")
            else:
                axs[subplot_idx, 1].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 1].set_xlabel(feature)
                axs[subplot_idx, 1].set_ylabel("Target")
        plt.savefig("linearity_scatter_plots.png")
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

# --> 6. VERIFICA SE PUÒ ESISTERE UNA CORRELAZIONE LINEARE TRA I DATI
    # Next we need to check if the data are co-linear. In linear regression high co-linearity between the features is a
    # problem. We can see how much the data is correlated by looking a correlation coefficient. Since our features are all
    # numerical, we'll use the famous Pearson correlation coefficient = -1 o +1 se E una correlazione lineare, 0 se non
    # esiste nessuna correlazione lineare!.
    if visualize:
        correlation_matrix = boston_pd.corr().round(2) # round sono le cifre decimali
        # annot = True to print the values inside the square
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        sns.heatmap(data=correlation_matrix, annot=True, ax=ax)
        plt.show()
        plt.close()
        # time.sleep(1)

# --> 7. VISUALIZZA LA CORRELAZIONE LINEARE TRA I DATI CHE HAI
    # DEVE rispettare condizioni slide 5/10/21!
    # Next, we create a correlation matrix that measures the linear relationships between
    # the variables. The correlation matrix can be formed by using the corr function from
    # the pandas dataframe library. We will use the heatmap function from the seaborn library
    # to plot the correlation matrix.
    if visualize:
        target = boston_pd["MEDV"]
        plt.figure(figsize=(20, 5))
        for i, col in enumerate(features): # FEATURES DIVENTA i, nome_Colonna
            plt.subplot(1, len(features), i + 1)
            x = boston_pd[col]
            y = target
            # N.B: la dimension dei vettori x e y deve combaciare
            plt.scatter(x, y, marker="o") # scatter è il grafico che devi definire tu le x e le y
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel("MEDV")
        # plt.savefig('sel_features_analysis.png')
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    X = boston_pd[features]     # [506 x 2 ] ["LSTAT", "RM"]
    Y = boston_pd[target_name]  # 506 [costo attuale dell'n-casa]

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
    plt.title("your LS solution")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    #plot3d_lr(W, X, "LS LR")
    return W, Y_pred


# SKLEARN Linear Regression model
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


# Gradient Descent
def gradfn(theta, X, Y):
    """
    Given `theta` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Return gradient of each weight evaluated at the current value
    """
    N, D = np.shape(X)
    Y_pred = np.matmul(X, theta) # X (506 x 2) theta deve essere 2x1 ris Y_pred = 506x1
    error = Y_pred - Y # ATT DEVE necessariamente essere così!!! sempre 506 x 1
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



def perform_lr_gd(X, Y, iters: int = 10000, alpha: float = 0.005):
    print(X.shape)
    theta = MyGDregression(X, Y, iters, alpha)
    Y_pred_GD = np.dot(theta[0:-1], X.T) + theta[-1] # di theta prendi solo gli N-1 valori!!
    loss_sgd = mean_squared_error(Y, Y_pred_GD)      # l'ultimo è il bias
    print("Model performance GD:")
    print("MSE is {}".format(loss_sgd))

    plt.figure(figsize=(4, 3))
    plt.scatter(Y, Y_pred_GD)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("your GD(my GDregression + update f)")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    #plot3d_lr(theta, X, "GD")

    return theta, Y_pred_GD


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

# HAI DECISO DI APPLICATE I METODI A 2 FEATURES features = ["LSTAT", "RM"]
# Data non-normalized, you have to do the linear regression least square
W, Y_pred = perform_lr_ls(X, Y)
print("R2 score(lr_ls)", r2_score(Y, Y_pred)) # residual_analisys(Y, Y_pred)

W, Y_pred = perform_lr_sklearn(X, Y)
# residual_analisys(Y, Y_pred)
print("R2 score (ls_sklearn)", r2_score(Y, Y_pred))

T, Y_pred = perform_lr_gd(X, Y) # gradient decent for finding a linear regression
# residual_analisys(Y, Y_pred)
print("R2 score(GD)", r2_score(Y, Y_pred))


# la tua implementazione del GD è giusta se e solo se gli errori sono uguali
# Standardizing data
scaler = StandardScaler() # crei l'ogg
scaler.fit(X) # gli dai in pasto i dati
X = scaler.transform(X) # ti salvi la trasformazione dei dati scalati

# Data normalized
W_norm, Y_pred_norm = perform_lr_ls(X, Y)
# residual_analisys(Y, Y_pred_norm)
print("R2 score Standardize data:", r2_score(Y, Y_pred_norm))

W_norm, Y_pred_norm = perform_lr_sklearn(X, Y)
# residual_analisys(Y, Y_pred_norm)
print("R2 score Standardize data:", r2_score(Y, Y_pred_norm))

T_norm, Y_pred_norm = perform_lr_gd(X, Y)
# residual_analisys(Y, Y_pred_norm)
print("R2 score Standardize data:", r2_score(Y, Y_pred_norm))

print("coef. dati non normalizzati:",T,"\ncoef. dati normalizzati:", T_norm)
print("!! I theta trovati prima (dove ultimo dato è bias) devono essere "
      "diversi perchè come training sono stati utilizzati dati diversi!!")

# ----------------------- LS vs GD -----------------------

# What if using an high number of features?
X = boston_pd[boston_raw.feature_names] # in questo caso prendi tutte le features a disposizione
# non viene controllato o meno se in ogni feature è possibile applicare o meno linear regressor....bha...

# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

print(f"Linear Regression on data: {X.shape}")

start = time.time()
W = LSRegression(X, Y)
print(f"durata applico mia LS regressor: {time.time() - start:.6f} seconds")

n_iter = 100
start = time.time()
clf_ = SGDRegressor(max_iter=n_iter)
clf_.fit(X, Y)
print(f"durata applico mio GD regressor: {time.time() - start:.6f} seconds")

# Ok maybe 13 features are still too few? Let's try something a little bigger, maybe 2000 features?
from sklearn.datasets import make_regression

X, Y = make_regression(n_samples=10000, n_features=2000)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

print(f"Linear Regression on super data: {X.shape}")

start = time.time()
W = LSRegression(X, Y)
print(f"Elapsed: {time.time() - start:.6f} seconds")

n_iter = 100
start = time.time()
clf_ = SGDRegressor(max_iter=n_iter)
clf_.fit(X, Y)
print(f"Elapsed: {time.time() - start:.6f} seconds")

timings_ls = []
timings_gd = []
feature_sizes = []
n_iter = 50
plt.figure(figsize=(10, 8))

for n_feat in range(100, 2000, 100):
    # n_samples = n_feat * 5
    n_samples = 10000
    n_features = n_feat
    X, Y = make_regression(n_samples=n_samples, n_features=n_features)

    feature_sizes.append(f"(S: {n_samples}, F: {n_features})")

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    print(f"Linear Regression on data: {X.shape}")

    start = time.time()
    W = LSRegression(X, Y)
    timings_ls.append(time.time() - start)

    clf_ = SGDRegressor(max_iter=n_iter)
    start = time.time()
    clf_.fit(X, Y)
    timings_gd.append(time.time() - start)

    plt.clf()
    x_axis = np.arange(len(timings_ls))
    plt.plot(x_axis, timings_ls, label="LS")
    plt.plot(x_axis, timings_gd, label="GD")
    plt.xticks(x_axis, feature_sizes, rotation=45)
    # Pad margins so that markers don't get clipped by the axes
    # plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.20)

    plt.ylabel("Execution time (s)")
    plt.xlabel("Runs")
    plt.grid()
    plt.legend()
    plt.draw()
    plt.pause(0.05)
