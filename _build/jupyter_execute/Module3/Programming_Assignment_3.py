#!/usr/bin/env python
# coding: utf-8

# # Week 3 Programming Assignment 
# 
# Remark: 
# 
# Please upload your solutions of this assignment to Canvas with a file named "Programming_Assignment_3 _yourname.ipynb" before deadline.

# =================================================================================================================

# ## **Problem 1 .** Use stochastic gradient descent method to train MNIST with 1 hidden layer neural network model to achieve at least 97% test accuracy. Print the results with the following format:
# 
#    "Epoch: i, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch.

# In[1]:


# write your code for solving probelm 1 in this cell


# =================================================================================================================

# ## **Problem 2 .** Use stochastic gradient descent method to train CIFAR-10 with
# * (1) logistic regression model to achieve at least 25% test accuracy 
# * (2) 2-hidden layers neural network model to achieve at least 50% test accuracy
# 
# Print the results with the following format:
# 
# * For logistic regression model, print:
# 
#     "Logistic Regression Model, Epoch: i, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# 
# * For 2-hidden layers neural network model, print:
# 
#     "DNN Model, Epoch: i, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# 
# where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch.
# 
# Hint: 
# 
# (1) The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
# 
# (2) The input_size should be $3072=3*32*32$, where 3 is the number of channels (RGB image), $32*32$ is the size of every image. 
# 
# (3) For the 2-hidden layers neural network model, consider to use $W^1\in \mathbb{R}^{3072\times3072}$ for the 1st-hidden layer, $W^2 \in \mathbb{R}^{500\times 3072}$ for the 2nd-hidden layer and $W^3 \in \mathbb{R}^{10\times 500}$ for the output layer.
# 

# In[2]:


# write your code for solving probelm 2 in this cell

# You can load CIFAR-10 dataset as follows:
CIFAR10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR10_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=CIFAR10_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


# =================================================================================================================
