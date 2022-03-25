#!/usr/bin/env python
# coding: utf-8

# # Week 5 Programming Assignment 
# 
# Remark: 
# 
# Please upload your solutions of this assignment to Canvas with a file named "Programming_Assignment_5 _yourname.ipynb" before deadline.

# =================================================================================================================

# ## Problem 1 : Use stochastic gradient descent method to train CIFAR10 with ResNet18.
# 
# Apply the following two different learning rate schedules respectively
# 
# (1) run 120 epochs with a fixed learning rate 0.1,
# 
# (2) run 120 epochs, and divide the learning rate by 10 every 30 epochs, which can achieve 94% test accuracy.
# 
# and print the results with the following format:
# 
#    "Epoch: i, Learning rate: lr$_i$, Training accuracy: $a_i$, Test accuracy: $b_i$"
# 
# where $i=1,2,3,...$ means the $i$-th epoch,  $a_i$ and $b_i$ are the training accuracy and test accuracy computed at the end of $i$-th epoch, and lr$_i$ is the learning rate of $i$-th epoch.
# 
# Optional Problem: Try to find some other learning rate schedules to achieve $94$% or higher test accuracy with less epochs.

# In[1]:


# Write your code to solve Problem 1.


# =================================================================================================================

# ## Problem 2 :  Consider the possion equation 
# \begin{equation}\label{1Dposi}
# \left\{
# \begin{aligned}
# -u''&= f, \,\, 0<x<1, \\
#  u(0)&=u(1)=0.
# \end{aligned}
# \right.
# \end{equation}
# 
# Assume $f(x)=1$, then the exact solution is $u(x)=\frac{1}{2}x(1-x)$. Given the partition with the grid points 
# $x_i=\frac{i}{n+1}, i=0,1,\cdots,n+1$, then by finite element discretization, 
# we obtain 
# 
# \begin{equation}\label{matrix}
# A\ast \mu =b,~~ A=\frac{1}{h}[-1,2,-1].
# \end{equation}
# 
# Use gradient descent method to solve the above problem with random initial guess $\mu^0$:
# 
# $$
# \mu^{m} = \mu^{m-1} - \eta ( A* \mu^{m-1}- b),~~ m=1,2,3,...,M
# $$
# 
# Set $n=15$ and $M=10$.
#     
# (1) Plot the curves of $u$ and $\mu^{M}$.
# 
# (2) Compute the error of the residual by $e^m = \sqrt{\sum_{i=0}^{n+1}|(A* \mu^{m}- b)_i |^2},~~ m=1,2,3,...,M$ and the index $i$ means the $i$-th entry of the vector. Plot a curve, where x-axis is $m=1,2,3,...,M$ and y-axis is $e^m$.
# 
# (3) Find the minumum $M$ when  $e^M = \sqrt{\sum_{i=0}^{n+1}|(A* \mu^{M}- b)_i |^2}<10^{-4}$ and record the computational time cost. 

# In[2]:


import numpy as np
import time
import matplotlib.pyplot as plt

######## FEM_GD  ########
# Write your code in FEM_GD function to compute one gradient descent iteration
def FEM_GD():


######## parameter definition ########
J = 4                                # grid level
n = 2**J - 1                         # number of inner grid points
h = 1/ 2**J                          # length of grid interval
x = np.arange(1, n + 1) *h           # grid points
u = 1/2*x*(1-x)                      # true solution at grid points
b = np.ones(n)*h                     # right-hand-size term
u1 = (np.random.rand(n)*2-1+np.sin(4*np.pi*x))/2  # initial value for u
M = 10
t0 = time.time()                     # initial time

######## compute numerical solution ########
err = []                             # create a list to save the error of each iteration
for m in range(M):
  u1 = FEM_GD({Define a FEM_GD function to compute one gradient descent iteration})
  temp=np.array([u1[0]*A[1]+u1[1]*A[2]])
  for j in np.arange(1,len(u1)-1):
      temp=np.append(temp,np.dot(u1[j-1:j+2],A))
  Au=np.append(temp,u1[-2]*A[0]+u1[-1]*A[1])
  err.append(np.linalg.norm(Au-b))   # compute the error of m-th iteration and save it to the list
print('time cost', time.time() - t0)
######## plot the exact solution and numerical solution ########
plt.figure()
plt.title('Exact solution and numerical solution')
plot = plt.plot(x,u,label='Exact solution')
plot = plt.plot(x,u1,'*',label='Numerical solution')
plt.legend()
plt.show()

######## plot the l2 norm of the error vs iterations ########
plt.figure()
plt.title('Error vs number of iterations using FEM and gradient descent')
plot = plt.plot(err)
plt.xlabel('Number of iterations')
plt.yscale('log')
plt.ylabel('Error')
plt.show()


# =================================================================================================================

# ## Problem 3 : Consider the Poisson equation described in Problem 1, call the Multigrid code given in the following cell to obtain a solution $u^{\nu}$.
# Use multigrid method to solve the above problem with random initial guess $\mu^0$:
# 
# $$
# \mu^{m} =  MG1(\mu^{m-1}),~~ m=1,2,3,...,M
# $$
# 
# Set $n=15$ and $M=10$.
#     
# (1) Plot the curves of $u$ and $\mu^{M}$ and record the computational time cost.
# 
# (2) Compute the error of the residual by $e^m = \sqrt{\sum_{i=0}^{n+1}|(A* \mu^{m}- b)_i |^2},~~ m=1,2,3,...,M$ and the index $i$ means the $i$-th entry of the vector. Plot a curve, where x-axis is $m=1,2,3,...,M$ and y-axis is $e^m$.

# In[3]:


import numpy as np
import time
import matplotlib.pyplot as plt

######## MG1 definition ########
def MG1(b, u0, J, v):  # \mu=[\mu_1,\mu_2,\mu_3,...,\mu_J]
  if len(b)!=len(u0):
    print('input size not equal')
  if len(v)!=J:
    print('length of v not equal to J')
  B=[0,b]
  U=[0,u0]
  R=np.array([1/2,1,1/2])
  for l in np.arange(1,J+1):
    h_l=1/2**(J+1-l)
    A=np.array([-1,2,-1]/h_l)
    if l<J:
      for i in np.arange(0,v[l-1]):
        temp=np.array([U[l][0]*A[1]+U[l][1]*A[2]])
        for j in np.arange(1,len(U[l])-1):
            temp=np.append(temp,np.dot(U[l][j-1:j+2],A))
        temp=np.append(temp,U[l][-2]*A[0]+U[l][-1]*A[1])
        U[l]+=1/4*h_l*(B[l]-temp)
      U.append(np.zeros((len(U[l])-1)//2))
      newb=[]
      temp=np.array([U[l][0]*A[1]+U[l][1]*A[2]])
      for j in np.arange(1,len(U[l])-1):
          temp=np.append(temp,np.dot(U[l][j-1:j+2],A))
      temp=np.append(temp,U[l][-2]*A[0]+U[l][-1]*A[1])
      for k in range((len(U[l])-1)//2):
        newb.append(np.dot((B[l]-temp)[2*k:2*k+3],R))
      B.append(newb)
    else:
      for i in np.arange(0,v[l-1]):
        temp=np.array(U[l][0]*A[1])
        U[l]+=1/4*h_l*(B[l]-temp)

  for l in np.arange(J-1,0,-1):
    temp=[1/2*U[l+1][0]]
    for i in np.arange(1,len(U[l+1])*2):
      if i%2==1:
        temp.append(U[l+1][(i-1)//2])
      else:
        temp.append(1/2*(U[l+1][(i-2)//2]+U[l+1][i//2]))
    temp.append(1/2*U[l+1][-1])
    U[l]+=temp
  return U[1]


######## parameter definition ########
J = 4                                # grid level
n = 2**J - 1                         # number of inner grid points
h = 1/ 2**J                          # length of grid interval
x = np.arange(1, n + 1) *h           # grid points
u = 1/2*x*(1-x)                      # true solution at grid points
b = np.ones(n)*h                     # right-hand-size term
u1 = np.random.rand(n)*2-1           # initial value for u
M = 10
t0 = time.time()

######## compute numerical solution ########
err = []                             # create a list to save the error of each iteration
{Write your code here to call MG1 function}
print('time cost', time.time() - t0)
######## plot the exact solution and numerical solution ########
plt.figure()
plt.title('Exact solution and numerical solution')
plot = plt.plot(x,u,label='Exact solution')
plot = plt.plot(x,u1,'*',label='Numerical solution')
plt.legend()
plt.show()

######## plot the l2 norm of the error vs iterations ########
plt.figure()
plt.title('Error vs number of iterations using multigrid')
plot = plt.plot(err)
plt.xlabel('Number of iterations')
plt.yscale('log')
plt.ylabel('Error')
plt.show()


# =================================================================================================================
