#!/usr/bin/env python
# coding: utf-8

# # Week 4 Programming Assignment 
# 
# Remark: 
# 
# Please upload your solutions of this assignment to Canvas with a file named "Programming_Assignment_4 _yourname.ipynb" 

# =================================================================================================================

# ### Problem 1 (5 pts): Given a image $x$, use Pytorch to apply the following three operations to the image sequentially:
# 
# ### (1) Do a convolution  $x_{conv1}= self.conv1(x)$ with $stride=1$,  zero $padding=2$ and  the $kernel_1$:
# ### $$
# Kernel_1=\begin{bmatrix} 
# 0 & 0 & -1 & 0 & 0 \\                 
# 0&-1&-2&-1& 0\\
# -1&-2&16&-2&-1\\
# 0&-1&-2&-1& 0\\
# 0& 0&-1& 0& 0 
# \end{bmatrix}
# $$
# 
# ### (2) Do a ReLu $x_{relu} = F.relu(x_{conv1})$
# 
# ### (3) Do another convolution  $x_{conv2}= self.conv2(x_{relu})$  with $stride=1$,  zero $padding=1$ and the average $kernel_2$:
# ### $$
# Kernel_2=\begin{bmatrix} 
# \frac{1}{9} & \frac{1}{9} &\frac{1}{9}  \\                 
# \frac{1}{9} & \frac{1}{9} &\frac{1}{9}  \\                 
# \frac{1}{9} & \frac{1}{9} &\frac{1}{9}  
# \end{bmatrix}
# $$
# ### Define a model which includes:  a convolutional layer self.conv1(), ReLu and another convolutional layer self.conv2(). Plot four images which are $x,~x_{conv1},~x_{relu},~x_{conv2}.$
# 

# In[1]:


# You can finish the following code to solve Problem 1.
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
#Step I: Deal with the data
def read_image():    
    im = Image.open('./lena.png')
    im_array = np.array(im)
    # transfer im_array to 4th order torch.tensor 
    im_array=torch.from_numpy(im_array)
    im_array=im_array.reshape(1,1,im_array.size(0),im_array.size(1))
    im_array=im_array.type(torch.FloatTensor)
    return im_array

#Step II: Define a function to plot the image: give 4th order torch.tensor 

def plot_images(images):
    plt.rcParams["figure.figsize"]=10,10 # change the figure size for plotting
    images_for_plot = images[0,0,:,:] 
    plt.imshow(images_for_plot.detach().numpy(), cmap='gray') 
    plt.show()
    

#Step III: Define the operators
    
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # Define the first convolutional layer
        kernel_1 = torch.tensor({Define your kernel here},dtype=torch.float32)
        # reshape your 2nd order tensor to 4th order tensor. Think about why?
        kernel_1 = kernel_1.reshape(1,1,kernel_1.size(0),kernel_1.size(1)) 
        self.conv1 = nn.Conv2d(1, 1, kernel_1.size(2), padding=2)
        # assign kerner_1 to conv1 layer.
        self.conv1.weight = torch.nn.Parameter(kernel_1)
        
        
        # Define the second convolutional layer
            
    def forward(self, x):
        x_conv1 = {Write your code}
        x_relu = {Write your code}
        x_conv2 = {Write your code}
        return x_conv1,x_relu,x_conv2
    
#Step IV: Show results:
x=read_image()
my_model=model()
x_conv1,x_relu,x_conv2 = my_model(x)

print('Original image')
{Write your code to plot the original image}
print('Apply the first convolution')
{Write your code to plot the image after applying the first convolution}
print('Apply the first convolution and ReLU')
{Write your code to plot the image after applying the first convolution and ReLU}
print('Apply the first convolution, ReLU and the second convolution')
{Write your code to plot the image after applying the first convolution, ReLU and the second convolution}


# =================================================================================================================

# ### **Problem 2 (5 pts).** Try to use stochastic gradient descent method to train CIFAR10 with LeNet-5 to achieve 60% test accuracy. Apply the following two strategies:
# 
# * ### Run 30 epochs, and divide the learning rate by 10 every 10 epochs
# * ### Weight decay 
# * ### Data normalization
# 
# 
# 
# ### Print the results with the following format:
# 
#    "Epoch: i, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch.

# In[ ]:





# =================================================================================================================

# ### **Optional Problem 1.** Try to use stochastic gradient descent method to train MNIST with LeNet-5 to achieve 99% test accuracy. Apply the following two strategies:
# 
# * ### Run 20 epochs, and divide the learning rate by 10 every 10 epochs
# * ### Weight decay 
# 
# ### Print the results with the following format:
# 
#    "Epoch: i, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch.

# =================================================================================================================

# ### **Optional Problem 2.** Try to use stochastic gradient descent method to train CIFAR10 with a CNN to achieve 70% test accuracy. For the CNN model, you can modified the LeNet-5:
# 
# * ### Increase the number of out_channels in conv1 from 6 to 16 or more.
# * ### Increase the number of out_channels in conv2 from 16 to 32 or more.  
# * ### You will also need to change the size of fc1 layer.
# 
# 
# Apply the following two strategies:
# 
# * ### Run 30 epochs, and divide the learning rate by 10 every 10 epochs
# * ### Weight decay 
# * ### Data normalization
# 
# 
# 
# ### Print the results with the following format:
# 
#    "Epoch: i, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch.
# 
# 

# In[ ]:




