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
# Which two sets are linearly separable ?
# 
# 

# ```{dropdown} Show answer
# Answer:  $A_1$,$A_3$ are linearly separable
# ```

# ## Question 2
# Consider three sets $A_1,A_2,A_3\subset R^d$, and three sentences listed below
# 
# -A: $A_1,A_2,A_3$ are all-vs-one linearly separable.
# 
# -B: $A_1,A_2,A_3$ are  linearly separable.
# 
# -C: $A_1,A_2,A_3$ are pairwise linearly separable.
# 
# Which sentence implies other sentence ? 

# ```{dropdown} Show answer
# Answer: A implies B; C implies B
# ```

# # Question 3 
# Let 
# 
# $$
#     \boldsymbol a=
# \begin{bmatrix}
# 1\\
# 1\\
# 1\\
# \vdots\\
# 1
# \end{bmatrix}
# \in R^n, A=\boldsymbol a\boldsymbol a^T
# $$
# 
# Compute the eigenvalues and corresponding eigenvectors of $A$.
# Write out solution

# ```{dropdown} Show answer
# $$
#     \lambda =n , x = \begin{bmatrix}
# 1\\
# 1\\
# 1\\
# \vdots \\
# 1
# \end{bmatrix}
# $$ 
# 
# $$
#     \lambda = 0,
#     x = \begin{bmatrix}1\\-1\\0\\0\\ \vdots\\0 \end{bmatrix}
#     \begin{bmatrix}1\\0\\-1\\0\\ \vdots\\0 \end{bmatrix}
#     \cdots
#     \begin{bmatrix}1\\0\\ \vdots\\0\\0\\-1\end{bmatrix}
# $$
# ```

# ## Question 4
# Determine the global minimizer and the global minimum of the function
#  $f(x,y)=e^{(x-2)^2+y^2+2x^2+2xy+2(y-4)^2+4y-16x} $

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
# 

# ```{dropdown} Show answer
# $\frac{1}{2}(A+A^T)$
# ```

# ## Question 6
# To define 
# 
# $$
#     A=
# \begin{bmatrix}
# 1&2&3\\
# 4&5&6\\
# 7&8&9
# \end{bmatrix}
# $$
# 
# what Python code should you use ? 
# 

# ```{dropdown} Show answer
#  A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# ```

# In[ ]:




