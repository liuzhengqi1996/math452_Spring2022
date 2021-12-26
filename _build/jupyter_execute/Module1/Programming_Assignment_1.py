#!/usr/bin/env python
# coding: utf-8

# # Module 1 Programming Assignment 
# 
# 
# Remark: 
# 
# (1) There are 9 problems, among which 7 problems are for 10 points whlie the other 2 problems are optional. Although the optional problems will not be counted, you are suggested to finish them as possible as you can.
# 
# (2) Please upload your solutions of this assignment to Canvas with a file named "Programming_Assignment_1 _yourname.ipynb" 

# =================================================================================================================

# ### **Problem 1 (1 pt).** Plot the curve of function 
# ### $$f(x) = x^3 - x,~~ x \in[-1,1].$$

# In[1]:


# write your code for solving probelm 1 in this cell


# =================================================================================================================

# ### **Problem 2 (2 pts).** 
# ### (1) Find the two roots of the function
# ### $$f(x) = 2x^2 - x - 1,~~ x \in[-1,2].$$
# 
# ### (2) Plot the curve of the function $f(x)$ defined in (1) and mark all the roots on the curve.

# =================================================================================================================

# ### **Problem 3 (optional)**.  Given $P=\begin{bmatrix} 1 & 2 \\ 3 & 4\end{bmatrix}$, investigate the two different multiplications $P*P$ and $torch.mm(P,P)$.

# =================================================================================================================

# ### **Problem 4 (1 pt).**  Given 
# ### $$A=\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}, ~~ b=\begin{bmatrix} 1  \\ 2 \\ 3\end{bmatrix}.$$
# ### Solve $A^2 x = b.$

# =================================================================================================================

# ### **Problem 5 (1 pt).**  Given $f(x,y) = x^2+y^2+(xy)^3$, compute $\frac{\partial f(x,y)}{\partial y}$ at $x=1,y=2$.

# =================================================================================================================

# ### **Problem 6 (optional).** Define a function to find the minimum of three numbers $a,~b,~c$. Test your code and print the minimum of the three numbers, where $a=\sqrt{2},~ b=\frac{4}{3},~ c=0.5e$.

# =================================================================================================================

# 
# ### **Problem 7 (1 pt).** Define a function to find the maximum and minimum of a sequence with n numbers.
# 
# Hint: x = np.random.randint(a,b,size=n) can randomly generate n numbers (saved in a row vector x ) and each number is between a and b.
# 

# =================================================================================================================

# ### **Problem 8 (1 pt).**  Define a function to sort a sequence with n numbers in ascending order.
# 

# =================================================================================================================

# ### **Problem 9 (3 pts).**  Given a function 
# ### $$f(x,y)=(x-2)^2 + y^2 + 2(y-4)^2 + 2x^2 + 2xy + 4y -16x +1.$$
# ### Please write a code to apply gradient descent method to find the minimum of $f(x,y)$ with initial value $x=y=0$.
# 
