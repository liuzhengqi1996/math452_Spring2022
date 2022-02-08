#!/usr/bin/env python
# coding: utf-8

# # Quiz 1

# In[1]:


import ipywidgets as widgets


# ## Question 1
# Consider three sets 
# 
# $$
#     A_1=\{(x,y)\in R^2:x^2+y^2\le \frac{1}{4}\}, \quad A_2=\{(x,y)\in R^2:(x-\frac{1}{2})^2+y^2\le \frac{1}{4}\}
# $$
# 
# and $A_3=\{(x,y)\in R^2:(x-\frac{3}{2})^2+y^2\le \frac{1}{4}\}$
# 
# Which of the following statement is correct? 
# 
# -I: $A_1$,$A_3$ are linearly separable
# 
# -II: $A_2$,$A_3$ are linearly separable
# 
# -III: $A_1$,$A_2$ are linearly separable
# 
# -IV: None of the above

# In[2]:


def on_change1(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'I':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w1 = widgets.RadioButtons(
    options=['I', 'II', 'III','IV'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w1.observe(on_change1)
w1


# ```{dropdown} Show answer
# Answer: I
# ```

# ## Question 2
# Consider three sets $A_1,A_2,A_3\subset R^d$, and three sentences listed below
# -A: $A_1,A_2,A_3$ are all-vs-one linearly separable.
# 
# -B: $A_1,A_2,A_3$ are  linearly separable.
# 
# -C: $A_1,A_2,A_3$ are pairwise linearly separable.
# 
# Which of the following statements is correct? 

# In[3]:


def on_change2(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'A implies B; C implies B':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w2 = widgets.RadioButtons(
    options=['A implies B; C implies B', 'B implies A; C implies B', 'B implies A; B implies C'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w2.observe(on_change1)
w2


# ```{dropdown} Show answer
# Answer: A implies B; C implies B
# ```

# # Question 3 
# Let $\boldsymbol a=
# \left (
# \begin{matrix}
# 1\\
# 1\\
# 1\\
# \vdots\\
# 1
# \end{matrix}
# \right)\in R^n, A=\boldsymbol a\boldsymbol a^T$. Compute the eigenvalues and corresponding eigenvectors of $A$.
# Write out solution

# ```{dropdown} Show answer
# 
# ```

# ## Question 4
# Determine the global minimizer and the global minimum of the function
# $f(x,y)=e^{(x-2)^2+y^2+2x^2+2xy+2(y-4)^2+4y-16x} \,\, .$

# In[4]:


def on_change4(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'Minimizer (3,1). Minimum 1':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w4 = widgets.RadioButtons(
    options=['Minimizer (3,1). Minimum 1', 'Minimizer (2,4). Minimum LaTeX: e^5', 'Minimizer (3,1). Minimum LaTeX: e','Minimizer (0,0). Minimum 1'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w4.observe(on_change4)
w4


# ```{dropdown} Show answer
# Minimizer (3,1). Minimum 1.
# 
# ```

# ## Question 5
# Let 
# 
# $$
#     \boldsymbol x\in R^3, A=\left[
# \begin{matrix}
# 1&2&3\\
# 0&4&1\\
# -1&-1&3
# \end{matrix}
# \right]\in R^{3\times 3}
# $$
# 
# Compute the Hessian matrix of the multivariable function $f(\boldsymbol x)=\frac{1}{2} \boldsymbol x^TA \boldsymbol x.$
# 
# -I: $\frac{1}{2}(A+A^T)$
# 
# -II: $A$
# 
# -III: $A+A^T$
# 
# -IV: $AA^T$

# In[5]:


def on_change5(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'I':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w4 = widgets.RadioButtons(
    options=['I','II','III','IV'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w4.observe(on_change4)
w4


# ```{dropdown} Show answer
# I
# ```

# ## Question 6
# To define $A=\left[
# \begin{matrix}
# 1&2&3\\
# 4&5&6\\
# 7&8&9
# \end{matrix}
# \right]$
# 
# which of the following Python code is correct? 
# 
# -I: A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# 
# -II: A = torch.tensor([1,2,3],[4,5,6],[7,8,9])
# 
# -III: A = torch.tensor([1,2,3;4,5,6;7,8,9])
# 
# -IV: A = torch.tensor((1,2,3),(4,5,6),(7,8,9))

# In[6]:


def on_change6(change):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == 'I':
            print( "Your answer: %s     is correct"% change['new'] )
        else:
            print( "Your answer: %s     is wrong"% change['new'] )

w6 = widgets.RadioButtons(
    options=['I','II','III','IV'],
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description='',
    disabled=False
)
w6.observe(on_change6)
w6


# ```{dropdown} Show answer
# I
# ```

# In[ ]:




