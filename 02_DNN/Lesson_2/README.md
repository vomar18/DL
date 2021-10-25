# Lesson 2: Regularization, Backpropagation and Fully Connected Neural Networks

In this lab lesson we will 

## Requirements
- System OS: any, but linux (ubuntu >= 18.04) is preferred
- Python3 (any version, I suggest 3.8, because with 3.9 I had some compatibility issue), and I suggest you to install Miniconda (https://docs.conda.io/en/latest/miniconda.html) for handling the python virtual environments.
- Python Packages:
    - pytorch (last version or at least 1.x; follow the instructions in: https://pytorch.org/get-started/locally/ )
    - numpy
    - sklearn
    - matplotlib

- Dev Tool: I suggest you Pycharm (both Professional Edition, included with UNIVR account for free, or Community Edition which is always free) or VSCode, depending on your preferences. Pycharm is less lightweight w.r.t. VSCode but provides a lot of useful tools for debugging. VSCode has a lot of useful extensions. They are both valid.

For this session you may want to access the GPU, but is not mandatory. All the code will also run on CPU.

# Linear Regression via Linear Neural network and Backpropagation from scratch
You will have to implement the backpropagation algorithm for a simple linear neural network with two weights. As a task we will use the Linear Regression for comparison with the lab lesson 1.
Then we will learn how to do the same with the pytorch library. (files 1 and 2)

# Multinomial logistic classifier
You will have to implement the multinomial logistic classifier as we already seen during the class. A small linear neural network will be proposed, and you need to perform the training. (file 3)

# Fully-Connected Neural Networks
We will see how to train a deep learning classifier based on a fully-connected neural network. (files 4 and 5)
