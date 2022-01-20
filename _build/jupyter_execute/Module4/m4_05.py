#!/usr/bin/env python
# coding: utf-8

# # Building and Training Convolutional Neural Networks (CNNs) with Pytorch

# This lecture includes:
# 
# 1. Build CNNs
# 2. Train MNIST with CNNs
# 3. Train CIFAR10 with CNNs
# 4. Improve the test accuracy
#     * Normalize the data
#     * Weight decay
#     * learning rate schedule

# ## 1. Build CNNs

# ### Convolutional Layer 

# In[1]:


import torch
import torch.nn as nn 

#stride default value: 1
#padding default vaule: 0
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)


# In[3]:


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.conv2 = nn.Conv2d(1, 2, 3)
        self.conv3 = nn.Conv2d(3, 2, 3)


my_model=model()
print(my_model.conv1.weight.size()) # (out_channels, in_channels, kernel_size, kernel_size)
print(my_model.conv2.weight.size()) # (out_channels, in_channels, kernel_size, kernel_size)
print(my_model.conv3.weight.size()) # (out_channels, in_channels, kernel_size, kernel_size)


# In[ ]:


x = torch.randn(1, 1, 4, 4) # batch_size=1, channel =1, image size =  4 * 4 

print(x)

print(my_model(x))


# ### Pooling 

# In[ ]:


import torch.nn.functional as F

out = F.max_pool2d(input, kernel_size)

out = F.avg_pool2d(input, kernel_size)


# In[108]:


x = torch.tensor([[[1,3,2,1],[1,3,2,1],[2,1,1,1],[3,5,1,1]]],dtype=float)
print(x)

max_x = F.max_pool2d(x,2)
print(max_x)

avg_x = F.avg_pool2d(x,2)
print(avg_x)


# ## 2. Train MNIST with CNNs

# In[6]:


import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

# Define a LeNet-5
# Note that we need to reshape MNIST imgaes 28*28 to 32*32
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        # out.size() = [batch_size, channels, size, size], -1 here means channels*size*size
        # out.view(out.size(0), -1) is similar to out.reshape(out.size(0), -1), but more efficient
        # Think about why we need to reshape the out?
        out = out.view(out.size(0), -1) 
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


minibatch_size = 128
num_epochs = 2
lr = 0.1

# Step 1: Define a model
my_model =model()
if use_cuda:
    my_model = my_model.cuda()

# Step 2: Define a loss function and training algorithm
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=lr)


# Step 3: load dataset

MNIST_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                  torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train= True, download=True, transform=MNIST_transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size)

testset = torchvision.datasets.MNIST(root='./data', train= False, download=True, transform=MNIST_transform)


testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset)) 



#Step 4: Train the NNs
# One epoch is when an entire dataset is passed through the neural network only once.
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

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
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()  
        outputs = my_model(images)
        p_max, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    training_accuracy = float(correct)/total

    
    # Test accuracy
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(testloader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outputs = my_model(images)
        p_max, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_accuracy = float(correct)/total
        
    print('Epoch: {}, the training accuracy: {}, the test accuracy: {}' .format(epoch+1,training_accuracy,test_accuracy))  


# ## 3. Train CIFAR10 with CNNs

# In[8]:


import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # change the input channels from 1 to 3
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        # out.size() = [batch_size, channels, size, size], -1 here means channels*size*size
        # out.view(out.size(0), -1) is similar to out.reshape(out.size(0), -1), but more efficient
        # Think about why we need to reshape the out?
        out = out.view(out.size(0), -1) 
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

minibatch_size = 128
num_epochs = 2
lr = 0.1

# Step 1: Define a model
my_model =model()
if use_cuda:
    my_model = my_model.cuda()

# Step 2: Define a loss function and training algorithm
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=lr)


# Step 3: load dataset

CIFAR10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR10_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=CIFAR10_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Step 4: Train the NNs
# One epoch is when an entire dataset is passed through the neural network only once.
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        if use_cuda:
          images = images.cuda()
          labels = labels.cuda()

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
        if use_cuda:
          images = images.cuda()
          labels = labels.cuda()  
        outputs = my_model(images)
        p_max, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    training_accuracy = float(correct)/total

    
    # Test accuracy
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(testloader):
        if use_cuda:
          images = images.cuda()
          labels = labels.cuda()
        outputs = my_model(images)
        p_max, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_accuracy = float(correct)/total
        
    print('Epoch: {}, the training accuracy: {}, the test accuracy: {}' .format(epoch+1,training_accuracy,test_accuracy))  


# ## 4. Improve the test accuracy

# ### Normalize the data with the mean and standard deviation of the dataset
# 
# 
# $$ \tilde{x}[i,j,:,:] = \frac{x[i,j,:,:]-mean[j]}{std[j]},~~~~i=1,2,...,60000,~~~~j=1,2,3$$.

# In[10]:



CIFAR10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR10_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=CIFAR10_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


# ### Weight decay

# Define the loss function with $\ell_2$ regularization:
# $$L(\theta) :=\frac{1}{N} \sum_{j=1}^N\ell(y_j, h(x_j; \theta)) +  + \lambda (\|\theta\|_2^2).$$
# 
# The parameter $\lambda$ is called "weight_decay" in Pytorch.
# 

# In[24]:


optimizer = optim.SGD(my_model.parameters(), lr=lr, weight_decay = 0.0001)
# weight_decay is usually small. Two suggested values: 0.0001, 0.00001


# ### Learning rate schedule

# In[46]:


def adjust_learning_rate(optimizer, epoch, init_lr):
    #lr = 1.0 / (epoch + 1)
    lr = init_lr * 0.1 ** (epoch // 10)  # epoch // 10, calculate the quotient 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# In[50]:


init_lr = 1
optimizer = optim.SGD(my_model.parameters(), lr=init_lr, weight_decay = 0.0001)
num_epochs = 30
init_lr = 1
for epoch in range(num_epochs):
    current_lr = adjust_learning_rate(optimizer, epoch, init_lr)
    print('Epoch: {}, Learning rate: {}'.format(epoch+1,current_lr))


# # Reading material
# 
# 1. LeNet-5: https://engmrk.com/lenet-5-a-classic-cnn-architecture/
# 2. torch.nn.Conv2d: https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
# 3. Understand Convolutions:
# https://medium.com/apache-mxnet/convolutions-explained-with-ms-excel-465d6649831c#f17e
# https://medium.com/apache-mxnet/multi-channel-convolutions-explained-with-ms-excel-9bbf8eb77108
# https://gfycat.com/plasticmenacingdegu

# ## (Optional material) How to compute the mean and standard deviation of CIFAR10 dataset?

# In[17]:


import numpy as np
CIFAR10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR10_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)

mean = 0.
std = 0.
for i, (images, labels) in enumerate(trainloader):
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean = images.mean(2).sum(0)
    std = images.std(2).sum(0)

mean /= len(trainloader.dataset)
std /= len(trainloader.dataset)
print('mean:', mean.numpy())
print('std1:', std.numpy())


# In[ ]:




