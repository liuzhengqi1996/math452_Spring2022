#!/usr/bin/env python
# coding: utf-8

# # Building and Training Deep Neural Networks (DNNs) with Pytorch
# 
# 

# This lecture includes:
# 
# 1. Build DNNs
# 3. Train MNIST with DNNs

# ## Class in Python

# Classes in Python provide a means of data and functionality together

# In[1]:


# Define a class
class Math497():
    def __init__(self):
        self.num_students = 40
        num = 100
    def add_students(self,x):
        self.num_students += x 
    
summer_course = Math497() # create a object

print('The number of students is:', summer_course.num_students)
summer_course.add_students(5)
print('The number of students is:', summer_course.num_students)
summer_course.add_students(-5)
print('The number of students is:', summer_course.num_students)


# In[2]:


# Define a child class of Math497()
class undergraduate(Math497):
    def __init__(self):
        super().__init__()  # super().__init__()  inherit parent's __init__() function.
        self.num_undergraduate_students = 34
    def add_undergraduate_students(self,x):
        self.num_students += x  # self.add_students(x) also works
        self.num_undergraduate_students += x 


# In[3]:


ug =undergraduate()
print('The number of undergraduate students in Math497 is:', ug.num_undergraduate_students)
print('The number of students in Math497 is:', ug.num_students)

ug.add_undergraduate_students(2)
print('The number of undergraduate students in Math497 is:', ug.num_undergraduate_students)
print('The number of students in Math497 is:', ug.num_students)


# ## Module in Python

# Consider a module to be the same as a code library. To create a module just save the code you want in a file with the file extension .py
# 
# See Course.py as example

# In[4]:


import Course

#  If our module changed, we would have to reload it with the following commands
import imp
imp.reload(Course)

my_class = Course.Math497()
print('The number of students in my class is:', my_class.num_students)


# ## 1. Build DNNs

# DNN includes:    
# 
# * input layer: given images $x$
#  
# * $l$-hidden layers: denote $x^{0}=x$
# $$ 
# \begin{eqnarray}
# &x^{1} = \sigma (x^0{W^0}^{T}+b^0), &&\text{first hidden layer}\\
# &x^{2} = \sigma (x^1{W^1}^{T}+b^1), &&\text{second hidden layer}\\
# &\vdots &&\\
# &x^{l} = \sigma (x^{l-1}{W^{l-1}}^{T}+b^{l-1}), &&l\text{-th hidden layer}\\
# \end{eqnarray}
# $$
# 
# * output layer: outputs$=(x^{l}{W^{l}}^{T}+b^{l})$

# In[4]:


import torch.nn as nn
import torch.nn.functional as F
# Note that: 
# (1)torch.nn.Module is a Class
# (2)torch.nn is a Module
# You can not import torch.nn.Modules
class model(nn.Module):  #
    def __init__(self,input_size,num_classes):
        super().__init__() 
        self.fc1 = nn.Linear(input_size, 500) 
        self.fc2 = nn.Linear(500, 250) 
        self.fc3 = nn.Linear(250, num_classes) 
    def forward(self, x): #Defines the computation performed at every call.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = 784
num_classes = 10
hidden_size = 500
my_model =model(input_size, num_classes)
print(my_model.fc1.weight.size())
print(my_model.fc1.bias.size())
print(my_model.fc2.weight.size())
print(my_model.fc2.bias.size())
print(my_model.fc3.weight.size())
print(my_model.fc3.bias.size())

# Question: When we call model(images), the forward(self,x) will run automatically. Why?  check __call__ 


# ## 2. Train a DNN model on MNIST

# In[1]:


import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F

# Define a 1-hidden layer neural network.
class model(nn.Module): 
    def __init__(self,input_size,hidden_size,num_classes):
        super().__init__() 
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes) 
    def forward(self, x): 
        x = x.reshape(x.size(0), input_size) # you can reshape the iamges here. 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



input_size = 784
hidden_size = 500
num_classes = 10

minibatch_size = 128
num_epochs = 2
lr = 0.1

# Step 1: Define a model
my_model =model(input_size,hidden_size, num_classes)

# Step 2: Define a loss function and training algorithm
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=lr)


# Step 3: load dataset

MNIST_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train= True, download=True, transform=MNIST_transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size)

testset = torchvision.datasets.MNIST(root='./data', train= False, download=True, transform=MNIST_transform)


testloader = torch.utils.data.DataLoader(testset, batch_size=1) 

#Step 4: Train the NNs
# One epoch is when an entire dataset is passed through the neural network only once.
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        
        #images = images.reshape(images.size(0), 28*28) # move this reshape to model class

        # Forward pass to get the loss
        outputs = my_model(images) 
        loss = criterion(outputs, labels)
        
        # Backward and compute the gradient
        optimizer.zero_grad()
        loss.backward()  #backpropragation
        optimizer.step() #update the weights/parameters
        
    # Training accuracy
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        #images = images.reshape(images.size(0), 28*28) # move this reshape to model class
        outputs = my_model(images)
        p_max, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    training_accuracy = float(correct)/total

    
    # Test accuracy
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(testloader):
        #images = images.reshape(images.size(0), 28*28) # move this reshape to model class
        outputs = my_model(images)
        p_max, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_accuracy = float(correct)/total
        
    print('Epoch: {}, the training accuracy: {}, the test accuracy: {}' .format(epoch+1,training_accuracy,test_accuracy))               


# ## Reading material
# 
# 1. Details of torch.nn https://pytorch.org/docs/stable/nn.html
# 
# 2. Details of torch package https://pytorch.org/docs/stable/torch.html
# 

# In[ ]:




