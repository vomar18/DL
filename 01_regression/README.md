# Lesson 1: Linear Regression

In this lab lesson, you will see the actual implementation of what we have already seen during the class.
We will use the "Boston Housing Dataset". You can find here: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html the informations on labels and data structure.

## Requirements
- System OS: any, but linux (ubuntu >= 18.04) is preferred
- Python3 (any version, I suggest 3.8, because with 3.9 I had some compatibility issue), and I suggest you to install Miniconda (https://docs.conda.io/en/latest/miniconda.html) for handling the python virtual environments.
- Python Packages:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
- Dev Tool: Pycharm (both Professional Edition, included with UNIVR account for free, or Community Edition which is always free) or VSCode, depending on your preferences. Pycharm is less lightweight w.r.t. VSCode but provides a lot of useful tools for debugging.

For this session you will not need to access the GPU.

## Dataset EDA and Regularization
We firstly will inspect the dataset, in order to visualize how looks like the distribution of the data for the different features present in the dataset.
Then we will apply a data normalization algorithm in order to regularizate the data.


## Linear Regression: Closed-Form
As we said in the frontal lesson, we are going to compute the linear regression (LR) in the closed-form, by computing the formula shown during the class.

This will be done using the Least-Square (LS) algorithm.

## Linear Regression: SKLEARN
We will compare our Least Square solution with the Sklearn's LinearRegression function, and see that the result is the same.

## Linear Regression: Gradient Descent
Then we will implement the Gradient Descent algorithm in order to solve the LR problem.
We will discuss the peculiarities of this approach w.r.t. LS solution.

