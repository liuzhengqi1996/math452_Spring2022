#!/usr/bin/env python
# coding: utf-8

# # Quiz 2
# For Penn State student, access quiz [here](https://psu.instructure.com/courses/2177217/quizzes/4421196)

# In[1]:


import ipywidgets as widgets


# ## Question 1
# Consider $f(x,y)=e^{x^2+y^2}$ , compute the Hessian matrix and determine whether $f(x,y)$ is a convex function. 

# ```{dropdown} Show answer
# Answer:  Hessian matrix is 
# 
# $$
#     e^{x^2+y^2}
# \begin{bmatrix}
# 4x^2+2&4xy\\
# 4xy&4y^2+2
# \end{bmatrix}
# $$
# 
# ```

# ## Question 2
# Given any $w\in R^n,\:b\in R,\:$ consider the multivariable function $f(\boldsymbol x)=e^{\boldsymbol w\cdot \boldsymbol x+b}$  Whether $f(\boldsymbol x)$ is a convex function?

# ```{dropdown} Show answer
# Answer: Yes
# 
# ```

# ## Question 3
# Consider $f(x,y)=x^2.$ Whether $f(x,y)$ is a $\lambda-$ strongly convex function?

# ```{dropdown} Show answer
# Answer: No
# 
# ```

# ## Question 4
# Consider 
# 
# $$
#     f\left(
# \begin{matrix}
# x\\
# y
# \end{matrix}
# \right)=x^2+y^2
# $$
# 
# Given initial guess 
# 
# $$
#     \left(
# \begin{matrix}
# x^0\\
# y^0
# \end{matrix}
# \right)
# =
# \left(
# \begin{matrix}
# 1\\
# 2
# \end{matrix}
# \right), \eta=\frac14
# $$
# 
# , compute
# 
# two steps of the gradient  descent method for $f(x,y)$:
# 
# $$
#     \left(
# \begin{matrix}
# x^{k+1}\\
# y^{k+1}
# \end{matrix}
# \right)
# =
# \left(
# \begin{matrix}
# x^k\\
# y^k
# \end{matrix}
# \right)
# -
# \eta \nabla f\left(
# \begin{matrix}
# x^k\\
# y^k
# \end{matrix}
# \right), k=0, 1.
# $$

# ```{dropdown} Show answer
# Answer:  $\frac{1}{4},\frac{1}{2}$
# 
# ```

# ## Question 5
# Suppose a point $x$ is drawn at random uniformly from the square $[-1,1]\times[-1,1].$ Let 
# 
# $$
#     \boldsymbol v= \left(
# \begin{matrix}
# 1\\
# 1
# \end{matrix}
# \right)
# $$
# 
# and consider the random variable $\mathcal X_{\boldsymbol v} ={\boldsymbol x} \cdot {\boldsymbol v}$.  What are $\mathbb{E} [\mathcal X_{\boldsymbol v}]$ and  $\big(\mathbb{V}[ \mathcal X_{\boldsymbol v}]\big)^2$.

# ```{dropdown} Show answer
# Answer:
# 
# ```

# ## Question 6
# 

# In[2]:


def model(100,10):
   return nn.Linear(100,10)


# What are the sizes of W and b of the model?

# ```{dropdown} Show answer
# Answer: 
# Size of W: torch.Size([10, 100]), Size of b: torch.Size([10])
# ```

# ## Question 7

# Load MNIST dataset with batch_size=100 as follows

# In[ ]:


trainset = torchvision.datasets.MNIST(root='./data', train= True, download=True,transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

for i, (images, labels) in enumerate(trainloader):


# What are the sizes of variable images and labels?

# ```{dropdown} Show answer
# Answer: 
# 
# ```

# ## Question 8
# In the training process of MNIST dataset with mini-batch stochastic gradient descent(SGD) method, if we set bath_size = 600, how many iterations (or SGD steps) are there in one epoch?

# ```{dropdown} Show answer
# Answer: 
# 
# ```

# ## Question 9
# What is the output of the following code?
# 

# In[ ]:



sequence = torch.tensor(([[4,2,3],[1,5,6],[0,7,2]]))
maxvalue, index = torch.max(sequence, 1) 
print(maxvalue,',',index)


# ```{dropdown} Show answer
# Answer: 
# 
# ```

# ## Question 10
# What is the output of the following code? 

# In[6]:


num_correct = 0
labels = torch.tensor([1,2,3,4,0,0,0])
predicted = torch.tensor([0,2,3,4,1,2,0])
num_correct += (predicted == labels).sum()
print(num_correct)


# ```{dropdown} Show answer
# Answer: 
# 
# ```
