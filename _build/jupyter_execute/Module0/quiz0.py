#!/usr/bin/env python
# coding: utf-8

# # Preliminary Quiz

# In[1]:


import ipywidgets as widgets


# ## Queation 1
# ![image](./Quiz1_Question1.jpg)
# 
# Consider the feasible region given by the following inequalities. whose boundary lines are graphed above.
# 
# $$
#     x+2y \geq 6 \\
#     x+y \geq 4 \\
#     x \geq 0 \\
#     y \geq 0 \\
# $$
# 
# Which ONE of the following labels best indicate the feasible region described above?
# 

# In[2]:


def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'I':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w0 = widgets.RadioButtons(
    options=['I', 'II', 'III','IV','None'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w0.observe(on_change)


# In[3]:


w0


# ```{dropdown} Show answer
# Answer: III
# ```

# ## Question 2
# Let 
# 
# $$
#     f(x) = x_1^2+x_3^2+x_3^4, p =(\frac{1}{\sqrt{3}},\frac{1}{\sqrt{3}},\frac{1}{\sqrt{3}}) , g(t)=f(x+tp)
# $$
# 
# where $t$ is a scalar variable. Find $g^{'}(0)$ at $x=(1,1,1)$

# In[4]:


widgets.Dropdown(
    options=['3', '$\sqrt{3}$','1','0','$3\sqrt{3}$'],
    value='3',
    description='Number:',
    disabled=False,
)


# ```{dropdown} Show answer
# Answer: 
# 
# $$
#     3\sqrt{3}
# $$
# 
# ```

# ## Question 3
# Determine the global maximum and global minimum of the function
# 
# $$
#     f(x) = x^3-3x^2+5
# $$
#  on the interval $[-2,3]$

# In[5]:


widgets.IntText(
    value=0,
    description='Global Max:',
    disabled=False
)


# In[6]:


widgets.IntText(
    value=0,
    description='Global Min:',
    disabled=False
)


# ```{dropdown} Show answer
# Answer: 
# 
# 
#     Global maximum is 5; global minimum is -15.
# 
# ```

# ## Question 4
# Find the global minimizer and the global minimum value of the function 
# 
# $$
#     f(x,y)=(x-2)^2+y^2+2(y-4)^2+2x^2+2xy+4y-16x+1.
# $$
# 
# That is, determine the point $(x_0,y_0)$  which yields the global minimum value,  

# In[7]:


widgets.IntText(
    value=0,
    description='x_0 is :',
    disabled=False
)


# In[8]:


widgets.IntText(
    value=0,
    description='y_0 is :',
    disabled=False
)


# In[9]:


widgets.IntText(
    value=0,
    description='f_{min} is:',
    disabled=False
)


# ```{dropdown} Show answer
# Answer: 
# 
# 
# ```

# ## Problem 5
# Which of the following matrices has an inverse?
# 
# - A: 
# 
# $$
#     \begin{bmatrix}
# 3&4 \\
# 6&8 \\
# \end{bmatrix}
# $$
# 
# - B: 
# 
# $$
#     \begin{bmatrix}
# 0&-4 \\
# 0&10 \\
# \end{bmatrix}
# $$
# 
# - C: 
# 
# $$
#     \begin{bmatrix}
# 4&-10 \\
# 2&5 \\
# \end{bmatrix}
# $$
# 
# - D: 
# 
# $$
#     \begin{bmatrix}
# 1&4 \\
# 0&3 \\
# \end{bmatrix}
# $$
# 
# - E: 
# 
# $$
#     \begin{bmatrix}
# 0&0 \\
# 5&7 \\
# \end{bmatrix}
# $$
# 

# In[10]:


def on_change5(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'D':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w5 = widgets.RadioButtons(
    options=['A', 'B', 'C','D','E'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w5.observe(on_change5)
w5


# ```{dropdown} Show answer
# Answer: 
# D
# ```

# ## Question 6
# Let 
# 
# $$
#     A=\left(
# \begin{matrix}
# 1&-1&0\\
# -1&2&-1\\
# 0&-1&1
# \end{matrix}
# \right)
# $$
# 
# Compute the eigenvalues of $A$
# 
# 

# In[11]:


def on_change6(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == '1,0,3':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w6 = widgets.RadioButtons(
    options=['1,0,3', '1,-1,0', '1,-1','1,2,1','-1,2,-1'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w6.observe(on_change6)
w6


# ```{dropdown} Show answer
# Answer: 
# 
# 1,0,3
# ```

# ## Question 7
# Let 
# 
# $$
#     a=\left(
# \begin{matrix}
# 1\\
# 1\\
# 1\\
# 1
# \end{matrix}
# \right)\in R^4
# $$
# 
# and $A=aa^T$
# 
# Find the eigenvalues and corresponding eigenvectors of $A$
# 
# Write out solution, take a picture or scan it into a file (we prefer a pdf file) and upload from your computer.

# ```{dropdown} Show answer
# 
# ```

# In[ ]:




