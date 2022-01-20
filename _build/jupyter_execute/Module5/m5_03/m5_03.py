#!/usr/bin/env python
# coding: utf-8

# # Building and Training ResNet with Pytorch

# This lecture includes:
# 
# 1. Improve the test accuracy
#     * Normalize the data (D05_CNN.ipynb)
#     * Weight decay (D05_CNN.ipynb)
#     * learning rate schedule (D05_CNN.ipynb)
#     * Initialization
#     * Batch normalization 
# 2. Introduction to ResNet
# 3. Training ResNet18 on Cifar10

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_698w9wpq&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_2ynx11a7"  ,width='800', height='500')


# ## 1. Improve the test accuracy
# 

# ### Default initialization

#  The values of these weights are sampled from uniform distrubution $U(-\sqrt{k},\sqrt{k})$, where 
#  
#  $$k=\frac{1}{\text{in_channels*kernel_size*kernel_size}}$$ 
#  

# In[2]:


import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F

#stride default value: 1
#padding default vaule: 0
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
    
my_model=model()


print(my_model.conv1.weight.size()) # (out_channels, in_channels, kernel_size, kernel_size)
print(my_model.conv1.weight)


# ### Kaiming He's initialization

# In[3]:



nn.init.kaiming_uniform_(my_model.conv1.weight, nonlinearity='relu')

print(my_model.conv1.weight)


# ### Xavier's initialization

# In[4]:



nn.init.xavier_uniform_(my_model.conv1.weight,gain=nn.init.calculate_gain('relu'))
print(my_model.conv1.weight)


# ### Batch normalization

# In[5]:


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.bn1 =  nn.BatchNorm2d(5) # Apply bn1 to conv1 5, we need take the arguement be the out_channels of conv1
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

input = torch.randn(1, 1, 4, 4)
my_model=model()
print(input)
print(my_model(input))


# ## 2. ResNet

# In[6]:


'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

        # (1) If stride = 1, we are using the operations (6.32) in Algorithm 8 in 497Notes.pdf.
        #         the self.shortcut = nn.Sequential() does nothing to input data x
        # (2) If stride != 1, we are using the operations (6.33) in Algorithn 8 in 497Notes.pdf.
     
        #         in __init__() step: 
        #         the self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes)) 
        #         is equivalent to
        #         self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        #         self.bn3 = nn.BatchNorm2d(planes)
        
        #         in forward(self,x) step:
        #             self.shortcut(x) is equivalent to  self.bn3(self.conv3(x))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Initialization layer
        self.bn1 = nn.BatchNorm2d(64)   # Initialization layer
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # Layer 1
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)# Layer 2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)# Layer 3
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)# Layer 4
        self.linear = nn.Linear(512, num_classes) # Fully connected 

    # block helps to create an object of class BasicBlock
    # plannes: out_channels of this layer
    # num_blocks: how many basic blocks in this layer
    # stride: what's the stride of the first block in this layer
    # ResNet18:
    # Layer 1: num_blocks[0] = 2, strides = [1,1], planes = 64
    # Layer 2: num_blocks[1] = 2, strides = [2,1], planes = 128
    # Layer 3: num_blocks[2] = 2, strides = [2,1], planes = 256
    # Layer 4: num_blocks[3] = 2, strides = [2,1], planes = 512
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []  # create a list to save the blocks in this layer: layers=[block1,block2]
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes # the out_channels of previous block is the in_channels of next block
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # Initialization layer
        print('Initial:',out.size()) 
        out = self.layer1(out)                  # layer 1
        print('Apply layer1:',out.size())
        out = self.layer2(out)                  # layer 2
        print('Apply layer2:',out.size())       
        out = self.layer3(out)                  # layer 3
        print('Apply layer3:',out.size())       
        out = self.layer4(out)                  # layer 4
        print('Apply layer4:',out.size())
        out = F.avg_pool2d(out, 4)              # average pooling
        print('Apply average pooling:',out.size())
        out = out.view(out.size(0), -1)
        out = self.linear(out)                  # Fully connected 
        return out    
    
#ResNet18: 4 layers and each layer has 2 blocks
my_model = ResNet(BasicBlock, [2,2,2,2], num_classes=10) 

x = torch.randn(10,3,32,32)
print('Input:',x.size())
y = my_model(x)
print('Output:',y.size())


# ## 3. ResNet18 on Cifar10

# In order to obtain the state-of-arts results of ResNet18 on Cifar10, we use the following strategies:
# 
# 
# * We apply a standard data augmentation method for Cifar10, see the code Line 88-Line 102. 
# 
#     If you are interested in the data augmentation, please check detail in https://pytorch.org/docs/stable/torchvision/transforms.html
# 
# 
# 
# 
# * We apply SGD with momentum=0.9 as follows:
# 
#   optimizer = optim.SGD(my_model.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)
# 
#   If you are interested in the SGD with Momentum, please check detail in https://pytorch.org/docs/stable/optim.html
#   
#   
# 
# * Since we would like to fix the parameters in batch normlization during the computation of accuracy, we need to change the status of my_model.
#     
#     When training the model, we use "my_model.train()", see Line 113.
#     
#     When compute the accuracy, we use "my_model.eval()", see Line 129.
#     
#     For more detail, please watch the video "Batch normalization" in Week5's page.
#     
#     
#     
#     
# * In the computation of accuracy, we do not compute any gradient. Thus, we can use "with torch.no_grad():" to set require_grad = False for all the computation, see Line 133 and Line 148, which will reduce the memory usage and speed up computations.
# 
#     
#     
# 

# In[ ]:


import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from timeit import default_timer as timer


use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out    
    
def adjust_learning_rate(optimizer, epoch, init_lr):
    #lr = 1.0 / (epoch + 1)
    lr = init_lr * 0.1 ** (epoch // 30)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

minibatch_size = 128
num_epochs = 120
lr = 0.1

# Step 1: Define a model
my_model = ResNet(BasicBlock, [2,2,2,2], num_classes=10) #ResNet18

if use_cuda:
    my_model = my_model.cuda()

# Step 2: Define a loss function and training algorithm
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)


# Step 3: load dataset
normalize = torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.ToTensor(),
                                                  normalize])

transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=False)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

start = timer()

#Step 4: Train the NNs
# One epoch is when an entire dataset is passed through the neural network only once.
for epoch in range(num_epochs):
    current_lr = adjust_learning_rate(optimizer, epoch, lr)

    my_model.train()
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
    my_model.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        with torch.no_grad():
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
        with torch.no_grad():
          if use_cuda:
              images = images.cuda()
              labels = labels.cuda()
          outputs = my_model(images)
          p_max, predicted = torch.max(outputs, 1) 
          total += labels.size(0)
          correct += (predicted == labels).sum()
    test_accuracy = float(correct)/total
        
    print('Epoch: {}, learning rate: {}, the training accuracy: {}, the test accuracy: {}' .format(epoch+1,current_lr,training_accuracy,test_accuracy)) 

end = timer()
print('Total Computation Time:',end - start)


# # Reading material
# 
# 1. ResNet: https://arxiv.org/pdf/1512.03385.pdf
# 2. torch.nn.init: https://pytorch.org/docs/stable/nn.init.html?highlight=init
# 3. Details of torch.nn https://pytorch.org/docs/stable/nn.html

# In[ ]:




