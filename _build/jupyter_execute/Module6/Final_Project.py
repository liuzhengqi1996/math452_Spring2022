#!/usr/bin/env python
# coding: utf-8

# # MATH 497: Final Project
# 
# Remark: 
# 
# Please upload your solutions for this project to Canvas with a file named "Final_Project_yourname.ipynb".

# =================================================================================================================

# ## Problem 1 [20%]:  
# 
# Consider the following linear system
# 
# \begin{equation}\label{matrix}
# A\ast u =f,
# \end{equation}
# or equivalently $u=\arg\min \frac{1}{2} (A* v,v)_F-(f,v)_F$, where $(f,v)_F =\sum\limits_{i,j=1}^{n}f_{i,j}v_{i,j}$ is the Frobenius inner product.
# Here $\ast$ represents a convolution with one channel, stride one and zero padding one. The convolution kernel $A$ is given by
# $$ 
# A=\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix},~~
# $$
# the solution $ u \in \mathbb{R}^{n\times n} $, and the RHS $ f\in \mathbb{R}^{n\times n}$ is given by $f_{i,j}=\dfrac{1}{(n+1)^2}.$
# 
# 
# ### Tasks:
# Set $J=4$, $n=2^J-1$ and the number of iterations $M=100$. Use the gradient descent method and the multigrid method to solve the above problem with a random initial guess $u^0$. Let $u_{GD}$ and $u_{MG}$ denote the solutions obtained by gradient descent and multigrid respectively.
#     
# * [5%] Plot the surface of solution $u_{GD}$ and $u_{MG}$.
# 
# * [10%] Define error $e_{GD}^m = \|A * u^{m}_{GD}- f\|_F=\sqrt{\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{n} |(A * u^{m}_{GD}- f)_{i,j}}|^2 $ for $m=0,1,2,3,...,M$. Similarly, we define the multigrid error $e_{MG}^m$. Plot the errors $e_{GD}^m$ and $e_{MG}^m$ as a function of the iteration $m$ (your x-axis is $m$ and your y-axis is the error). Put both plots together in the same figure.
# 
# * [5%] Find the minimal $m_1$ for which $e^{m_1}_{GD} <10^{-5}$ and the minimal $m_2$ for which $e^{m_2}_{MG} <10^{-5}$, and report the computational time for each method. Note that $m_1$ or $m_2$ may be greater than $M=100$, in this case you will have to run more iterations.

# ### Remark:
# 
# Below are examples of using gradient descent and multigrid iterations for M-times 
# * #### For gradient descent method with $\eta=\frac{1}{8}$, you need to write a code:
# 
#     Given initial guess $u^0$
# $$
# \begin{align}
# &\text{for    }  m =  1,2,...,M\\
# &~~~~\text{for    }  i,j = 1: n\\
# &~~~~~~~~u_{i,j}^{m} = u_{i,j}^{m-1}-\eta(f_{i,j}-(A\ast u^{m-1})_{i,j})\\
# &~~~~\text{endfor}\\
# &\text{endfor}
# \end{align} 
# $$
# 
# * #### For multigrid method, we have provided the framework code in F02_MultigridandMgNet.ipynb:
# 
#     Given initial guess $u^0$
# $$
# \begin{align}
# &\text{for    }  m =  1,2,...,M\\
# &~~~~u^{m} = MG1(u^{m-1},f, J, \nu)\\
# &\text{endfor}
# \end{align} 
# $$

# =================================================================================================================

# ## Problem 2 [50%]: 
# 
# Use SGD with momentum and weight decay to train MgNet on the Cifar10 dataset. Use 120 epochs, set the initial learning rate to 0.1, momentum to 0.9, weight decay to 0.0005, and divide the learning rate by 10 every 30 epochs. (The code to do this has been provided.) Let $b_i$ denote the test accuracy of the model after $i$ epochs, and let $b^*$ = $\max_i(b_i)$ be the best test accuracy attained during training.
# 
# 
# ### Tasks:
#    * [30%] Train MgNet with the following three sets of hyper-parameters (As a reminder, the hyper-parameters of MgNet are $\nu$, the number of iterations of each layer, $c_u$, the number of channels for $u$, and $c_f$, the number of channels for $f$.):
#  
#     (1) $\nu=$[1,1,1,1], $c_u=c_f=64$.
#     
#     (2) $\nu=$[2,2,2,2], $c_u=c_f=64$.
# 
#     (3) $\nu=$[2,2,2,2], $c_u=c_f=64$, try to improve the test accuracy by implementing MgNet with $S^{l,i}$, which means different iterations in the same layer do not share the same $S^{l}$. 
#   
#   
#    * For each numerical experiment above, print the results with the following format:
# 
#        "Epoch: i, Learning rate: lr$_i$, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
#         where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch, and lr$_i$ is the learning rate of $i$-th epoch.
#     
#     
#    * [10%] For each numerical experiment above, plot the test accuracy against the epoch count, i.e. the x-axis is the number of epochs $i$ and y-axis is the test accuracy $b_i$. An example plot is shown in the next cell.
#    
#    
#    * [10%] Calculate the number of parameters that each of the above models has. Discuss why the number of parameters is different (or the same) for each of the models.
#        

# In[1]:


from IPython.display import Image
Image(filename='plot_sample_code.png')


# In[ ]:


# You can calculate the number of parameters of my_model by:
model_size = sum(param.numel() for param in my_model.parameters())


# =================================================================================================================

# ## Problem 3 [25 %]:
# 
# Try to improve the MgNet Accuracy by increasing the number of channels. (We use the same notation as in the previous problem.) Double the number of channels to $c_u=c_f=128$ and try different $\nu$ to maximize the test accuracy.
# 
# ### Tasks:
#    * [20%] Report $b^{*}$, $\nu$ and the number of parameters of your model for each of the experiments you run.
#    * [5%] For the best experiment, plot the test accuracy against the epoch count, i.e. the x-axis is the number of epochs $i$ and y-axis is the test accuracy $b_i$. (Same as for the previous problem.)

# In[ ]:


# You can calculate the number of parameters of my_model by:
model_size = sum(param.numel() for param in my_model.parameters())


# =================================================================================================================

# ## Problem 4 [5%]:
# 
# Continue testing larger MgNet models (i.e. increase the number of channels) to maximize the test accuracy. (Again, we use the same notation as in problem 2.)
# 
# ### Tasks:    
#     
# +  [5%] Try different training strategies and MgNet architectures with the goal of achieving $b^*>$ 95%. Hint: you can tune the number of epochs, the learning rate schedule, $c_u$, $c_f$, $\nu$, try different $S^{l,i}$ in the same layer $l$, etc...

# =================================================================================================================
