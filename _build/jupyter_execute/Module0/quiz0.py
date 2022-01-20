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


# In[4]:


import numpy as np
import nbinteract as nbi

def normal(mean, sd):
    '''Returns 1000 points drawn at random fron N(mean, sd)'''
    return np.random.normal(mean, sd, 1000)

# Pass in the `normal` function and let user change mean and sd.
# Whenever the user interacts with the sliders, the `normal` function
# is called and the returned data are plotted.
nbi.hist(normal, mean=(0, 10), sd=(0, 2.0))


# ```{dropdown} Show answer
# Answer: III
# ```

# In[5]:


w = widgets.Dropdown(
    options=['Addition', 'Multiplication', 'Subtraction', 'Division'],
    value='Addition',
    description='Task:',
)

def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        print("changed to %s" % change['new'])

w.observe(on_change)

display(w)


# In[6]:


w


# In[7]:


button = widgets.Button(
    description='Click me',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check' # (FontAwesome names without the `fa-` prefix)
)
button.on_click(on_change)
button


# ## Question 2
# Let 
# 
# $$
#     f(x) = x_1^2+x_3^2+x_3^4, p =(\frac{1}{\sqrt{3}},\frac{1}{\sqrt{3}},\frac{1}{\sqrt{3}}) , g(t)=f(x+tp)
# $$
# 
# where $t$ is a scalar variable. Find $g^{'}(0)$ at $x=(1,1,1)$

# In[8]:


widgets.Dropdown(
    options=['3', '$\sqrt{3}$','1','0','$3\sqrt{3}$'],
    value='3',
    description='Number:',
    disabled=False,
)


# ## Question 3
# Determine the global maximum and global minimum of the function
# 
# $$
#     f(x) = x^3-3x^2+5
# $$
#  on the interval $[-2,3]$

# In[9]:


widgets.IntText(
    value=0,
    description='Global Max:',
    disabled=False
)


# In[10]:


widgets.IntText(
    value=0,
    description='Global Min:',
    disabled=False
)


# In[ ]:




