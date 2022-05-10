#!/usr/bin/env python
# coding: utf-8

# # Quiz 5
# For Penn State student, access quiz [here](https://psu.instructure.com/courses/2177217)

# In[1]:


import ipywidgets as widgets


# ## Question 1
# Consider a DNN layer $f^\ell = W^\ell \sigma (f^{\ell-1}) + b^\ell$ , where $W^\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ with $n_\ell = n_{\ell-1} = m$. If we apply the Xavier's initialization for this layer, what is the suggested variance to sample $W_{st}^\ell$ ?
# 

# ```{dropdown} Show answer
# Answer:
# $\frac{1}{m}$
# ```

# ## Question 2
# When training a CNN model with batch normalization (BN) structure, let us consider the time step $t$ with mini-batch $\mathcal B_t$ for the $j$-th channel of $\ell$-th layer (spatial dimension (resolution) for this layer is $n_\ell \times m_\ell $).
# Then, what is the size for the commonly used mean $[\mu^\ell_{\mathcal B_t}]_j$ and variance 
# 	$[\sigma^\ell_{\mathcal B_t}]_j$  in BN for CNN models on this layer?

# ```{dropdown} Show answer
# Answer: $[\mu^\ell_{\mathcal B_t}]_j \in \mathbb{R}, [\sigma^\ell_{\mathcal B_t}]_j \in \mathbb{R}$
# ```

# ## Question 3
# If we define a convolutional layer with batch normalization as follows

# In[2]:


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.bn1 =  nn.BatchNorm2d(N)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))


# What is the value of N in nn.BatchNorm2d(N)?

# ```{dropdown} Show answer
# Answer: 10
# ```

# ## Question 4
# How many kernels/filters are there in the initialization layer self.conv1 of ResNet18?

# In[ ]:


self.conv1 = nn.Conv2d(3, 64, kernel_size=3, st
ride=1, padding=1, bias=False)


# ```{dropdown} Show answer
# Answer: 64
# ```

# ## Question 5
# What is the equivalent code of the following code?

# In[ ]:


Conv_BN = nn.Sequential(nn.Conv2d(1,3,3),nn.BatchNorm2d(3))
 
x = torch.randn(1, 1, 28, 28)

out = Conv_BN(x)


# ```{dropdown} Show answer
# Answer: Conv1 = nn.Conv2d(1,3,3)
# bn1 = nn.BatchNorm2d(3)
# x = torch.randn(1, 1, 28, 28)
# out = bn1(Conv1(x))
# ```

# ## Question 6
# In the following code, what is the size of out if the size of x is torch.Size([3, 3, 3, 3]) 

# In[ ]:


out = x.view(x.size(0), -1)


# ```{dropdown} Show answer
# Answer: torch.Size([3, 27])
# ```

# ## Question 7
# When we define ResNet18 as follows

# In[ ]:


my_model = ResNet(BasicBlock, [2,2,2,2], num_classes=10)


# what does [2,2,2,2] mean?

# ```{dropdown} Show answer
# Answer: There are 4 layers and each layer has 2 blocks
# ```

# ## Question 8
# Here, let $\sigma(x) = e^x, \quad x \in \mathbb{R}.$
# Consider the following 1-hidden layer DNN function with \sigma$ activation function for any $x\in \mathbb{R}^2$
# 
# $
# f(x;\theta) =  W^2 \sigma (W^1 x+ b^1)  \in \mathbb{R},
# $ 
# 
# where 
# 
# $\theta = \{ W^1, b^1, W^2\}$ and $W^1 \in \mathbb{R}^{2\times 2}, \quad W^2 \in \mathbb{R}^{1\times 2}, \quad b^1 \in \mathbb{R}^2.$ 
# 
# Calculate $\left. \frac{\partial f(x; \theta)}{\partial W^1_{st}} \right|_{\theta = \theta^*, x = x^*}
# 	\quad \text{and} \quad 
# 	\left. \frac{\partial f(x; \theta)}{\partial x_i} \right|_{\theta = \theta^*, x = x^*},$
#     
# for $i = 1,2$ and $s,t = 1,2$, where $\theta = \theta^*, x = x^*$ means 
# 
# $$
#     W^1 = 	
# 	\begin{pmatrix}
# 	0 & 1 \\
# 	1 & 0 
# 	\end{pmatrix},  
# 	W^2 = 	\begin{pmatrix}
# 	1 & 1
# 	\end{pmatrix}, b^1 = 
# 	\begin{pmatrix}
# 	0 \\ 0
# 	\end{pmatrix}
# $$
# 
# and
# 
# $$
#     x = \begin{pmatrix}
# 	1 \\0
# 	\end{pmatrix}
# $$

# ```{dropdown} Show answer
# Answer: Unavailable
# ```

# ## Question 9
# Consider the convolution for one channel with stride one and zero padding $A\ast: R^{n}\mapsto  R^{n}$.
# $
# A\ast u=f,
# $
# where 
# $
# A=\frac{1}{h}\begin{pmatrix}
# 	-1, &2,&-1
# 	\end{pmatrix}.
# $
# 
# Consider following two iterative methods for the above equation. 
# Given $u^{0}$, for $\ell=1,2,\cdots,2m$
# 
# $u^{\ell}=u^{\ell-1}+\frac{h}{4}(f-A\ast u^{\ell-1})$
# 
# And
# Given $\tilde{u}^{0}=u^{0}$, for $\ell=1,2,\cdots,m$
# 
# $\tilde{u}^{\ell}=\tilde{u}^{\ell-1}+S_1\ast(f-A\ast\tilde{u}^{\ell-1})$
# 
# Determine $S_1$ in the second iterative method such that $u^{2m}=\tilde{u}^{m} \quad\hbox{when}\quad m=1,$, namely $u^{2}=\tilde{u}^{1}$
# 

# ```{dropdown} Show answer
# Answer: Unavailable
# ```

# ## Question 10
# Consider the convolution for one channel with stride one and zero padding.
# Given $f\in \mathbb R^n$, let $u$ be the solution of the following linear system $A\ast u=f$,where $A=(-1,2,-1)$ 
# 
# (a) Show that the solution $u$ satisfies the minimization problem
# 
# (b) Write out the gradient descent method to solve the above minimization problem
# 

# ```{dropdown} Show answer
# Answer: Unavailable
# ```

# In[ ]:




