#!/usr/bin/env python
# coding: utf-8

# # Quiz 3
# For Penn State student, access quiz [here](https://psu.instructure.com/courses/2177217)

# In[1]:


import ipywidgets as widgets


# ## Question 1
# Is $f(x)=e^x$  a convex function? 

# ```{dropdown} Show answer
# Answer: Yes
# ```

# ## Question 2
# Consider the uniform distribution $\mathcal X$ on $[-a,a]$ for some number $a>0$. What are the expectation and variance of $\mathcal X$

# ```{dropdown} Show answer
# Answer:$0,   \frac{a^2}{3}.$
# ```

# ## Question 3
# Suppose you flip a fair icon 3 times. Let $\chi$ be the number of heads. Calculate the expectation of $\chi ^2 $

# ```{dropdown} Show answer
# Answer: 3
# ```

# ## Question 4
# Consider the function $f(x,y,z)=yz+e^{xyz}$. At the point 
# 
# $
#     \begin{pmatrix}
#     x\\
#     y\\
#     z
#     \end{pmatrix}
#     =
#     \begin{pmatrix}
#     0\\
#     1\\
#     2
#     \end{pmatrix}
# $
# 
# find the direction along which the function decreases most rapidly.
# 

# ```{dropdown} Show answer
# Answer: $\begin{pmatrix} -2\\-2\\-1\end{pmatrix}$
# ```

# ## Question 5
# Consider $f(x,y)=2x^2+2y^2.$ Given initial guess 
# 
# $
#     \begin{pmatrix}
#     x^0\\
#     y^0
#     \end{pmatrix}
#     =
#     \begin{pmatrix}
#     2\\
#     3
#     \end{pmatrix}
# $
# 
# $\eta =1/8$ 
# 
# compute two steps of the gradient  descent method for $f(x,y)$

# ```{dropdown} Show answer
# Answer: 
# 
# $
#     \begin{pmatrix}
#     x^2\\
#     y^2
#     \end{pmatrix}
#     =
#     \begin{pmatrix}
#     \frac {1}{2}\\
#     \frac {3}{4}
#     \end{pmatrix}
# $
# ```

# ## Question 6
# What is output of the following code?

# In[2]:


class test:
        def _ _init_ _(self, a):
               self.a=a
        def display(self):
               print(self.a)
obj = test()
obj.display()


# ```{dropdown} Show answer
# Answer: Error as one argument is required while creating the object
# ```

# ## Question 7
# If we use "import Course'' in Python, what is "Course"?

# ```{dropdown} Show answer
#     Answer: A module
# ```

# ## Question 8
# What is the output of the following code:

# In[ ]:


print('{}\n/{}'.format(1,2))


# ```{dropdown} Show answer
# Answer: 1
# 
# /2
# ```

# ## Question 9
# How to define stochastic gradient descent method with learing rate=1 after:

# In[ ]:


import torch.optim
import torch.nn as nn
my_model=nn.Linear(784,10)


# ```{dropdown} Show answer
# Answer: optimizer = torch.optim.SGD(my_model.parameters(), lr=1)
# ```

# ## Queation 10
# For MNIST dataset, if we would like to use full gradient descent method, how should we define the trainloader?

# ```{dropdown} Show answer
# Answer:trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000)
# ```

# In[ ]:




