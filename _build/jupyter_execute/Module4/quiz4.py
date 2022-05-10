#!/usr/bin/env python
# coding: utf-8

# # Quiz 4
# For Penn State student, access quiz [here](https://psu.instructure.com/courses/2177217)

# In[1]:


import ipywidgets as widgets


# ## Question 1
# Consider 
# 
# $$
#     g=\begin{pmatrix}
#     1 & 2 & 1 & 1\\
#     0 & 0 & 2 & 0\\
#     0 & 2 & 2 & 0\\
#     1 & 1 & 1 & 1
#     \end{pmatrix},
# $$
# and kernel
# 
# $$
#     K = 
#     \begin{pmatrix}
#     0 & 0 & 1\\
#     0 & 1 & 0\\
#     1 & 0 & 0
#     \end{pmatrix}.
# $$
# 
# Then what is the outcome of convolution for one channel 
# 
# with stride one and zero padding $ f = K\ast g $
# 

# ```{dropdown} Show answer
# Answer:
# $
#     f = \begin{pmatrix}
#     1 & 2 & 1 & 3\\
#     2 & 1 & 5 & 2\\
#     0 & 5 & 3 & 1\\
#     3 & 3 & 1 & 1
#     \end{pmatrix}
# $
# ```

# ## Question 2
# How many parameters are needed when using a 3x3 convolutional operations with 2 input and 5 output channels?

# ```{dropdown} Show answer
# Answer: 90
# ```

# ## Question 3
# Consider a two channel image (tensor) $g \in \mathbb{R}^{2\times 5\times 5}$. Let define $f = K\ast g$ by convolution for multi-channel with stride-two and zero padding, where $K \in \mathbb{R}^{3\times 2\times 3\times3}$ ($K_{i,j} \in \mathbb{R}^{3\times 3}$ for $i=1,2,3$ and $j=1,2$). What is the right dimension (size) of $f$.

# ```{dropdown} Show answer
# Answer: $f \in \mathbb{R}^{3\times 3\times 3}$
# ```

# ## Question 4
# Given 
# 
# $$
#     \tilde{M}=
#     \left(
#     \begin{array}{ccccc}
#     4& 1&&&\\
#     1&4&1&&\\
#     &\ddots&\ddots&\ddots&\\
#     &&1&4&1\\
#     &&&1&4\\
#     \end{array}
#     \right)\in \mathbb R^{n\times n}
# $$
# 
# Write out the kernel $M$ such that for any $\mu\in \mathbb R^n$ (with zero padding)
# $\tilde{M}\mu=M\ast \mu.$

# ```{dropdown} Show answer
# Answer: $M=(1,4,1)$
# ```

# ## Question 5
# Consider a mesh $\mathcal T_1$ of $[0, 1]$ with mesh size $h_1=\frac14$, namely $[0,1]=[0,\frac14]\cup [\frac14,\frac12]\cup [\frac12, \frac34]\cup[\frac34,1]$ and the corresponding basis function 
#  $\phi^1_1(x), \phi^1_2(x), \phi^1_3(x)$. 
#  Next we consider another mesh $\mathcal T_2$ of $[0, 1]$ with mesh size $h_2=\frac12$, namely 
#  $[0, 1]=[0,\frac12]\cup [\frac12,1]$ and the corresponding basis function $\phi_1^2(x)$.
#  Find $a_1,a_2,a_3$ such that 
#  $
#  \phi_1^2(x)=a_1\phi_1^1(x)+a_2\phi_2^1(x)+a_3\phi_3^1(x).
#  $

# ```{dropdown} Show answer
# Answer: $(a_1,a_2,a_3)=(\frac{1}{2},1,\frac{1}{2})$
# ```

# ## Question 6
# What is output of the following code?

# In[2]:


conv1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)

print(conv1.weight.size())


# ```{dropdown} Show answer
# Answer: torch.Size([2, 1, 3, 3])
# ```

# ## Question 7
# What is the output of the following code:

# In[ ]:


import torch.nn.functional as F

x = torch.tensor([[[1,1,2,2],[0,0,0,0],[3,3,4,4],[0,0,0,0]]],dtype=float)

max_x = F.max_pool2d(x,2)

avg_x = F.avg_pool2d(x,2)


print(max_x,'\n',avg_x)


# ```{dropdown} Show answer
# Answer:tensor([[[1.0, 2.0],[3.0, 4.0]]])
# tensor([[[0.5, 1.0],[1.5, 2.0]]])
# ```

# ## Question 8
# What is the output of the following code:

# In[ ]:


x = torch.randn(1, 3, 4, 4)

x = x.view(x.size(0), -1) 

print(x.size())


# ```{dropdown} Show answer
# Answer: torch.Size([1, 48])
# ```

# ## Question 9
# If we ask that the learning rate is divided by 5 every 30 epochs, given the initial learning is 1, what code should we run?

# ```{dropdown} Show answer
# Answer: lr = 1 * 0.2 ** (epoch // 30)
# ```
