#!/usr/bin/env python
# coding: utf-8

# # Optimization and gradient descent method

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_wota11ay&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_o38cisoq",width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2021/12/A06GradientDescent_Video_Notes.pdf)

# ## Gradient descent method
# 
# For simplicity, let us just consider a general optimization problem
# 
# $$
#     \label{optmodel}
#     \min_{x\in \mathbb{R}^n } f(x).
# $$ (problem)
# 
# ![image](../figures/diag_GD.png)
# 
# ### A general approach: line search method
# 
# Given any initial guess $x_1$, the line search method uses the following
# algorithm
# 
# $$
#     \eta_t= argmin_{\eta\in \mathbb{R}^1} f(x_t - \eta p_t)\qquad \mbox{(1D minimization problem)}
# $$
# 
# to produce $\{ x_{t}\}_{t=1}^{\infty}$ 
# 
# $$
#     x_{t+1} = x_{t} - \eta_t p_t.
# $$ (line-search)
# 
# Here $\eta_t$ is called the step size in
# optimization and also learning rate in machine learn
# ing, $p_t$ is called
# the descent direction, which is the critical component of this
# algorithm. And $x_t$ tends to
# 
# $$
#     x^*= argmin_{x\in \mathbb{R}^n} f(x) \iff f(x^*)=\min_{x\in \mathbb{R}^n} f(x)
# $$
# 
# as $t$ tends to infinity. There is a series of optimization algorithms
# which follow the above form just using different choices of $p_t$.
# 
# Then, the next natural question is what a good choice of $p_t$ is? We
# have the following theorem to show why gradient direction is a good
# choice for $p_t$.
# 
# ```{admonition} lemma
# Given $x \in \mathbb{R}^n$, if $\nabla f(x)\neq 0$, the fast descent
# direction of $f$ at $x$ is the negative gradient direction, namely
# 
# $$
#     -\frac{\nabla f(x)}{\|\nabla f(x)\|} = \mathop{\arg\min}_{ p \in \mathbb{R}^n, \|p\|=1} \left. \frac{\partial f(x + \eta p)}{\partial \eta} \right|_{\eta=0}.
# $$
# 
# It means that $f(x)$ decreases most rapidly along the negative gradient
# direction.
# ```
# 
# ```{admonition} proof
# *Proof.* Let $p$ be a direction in $\mathbb{R}^{n},\|p\|=1$. Consider
# the local decrease of the function $f(\cdot)$ along direction $p$
# 
# $$
#     \Delta(p)=\lim _{\eta \downarrow 0} \frac{1}{\eta}\left(f(x+\eta p)-f(x)\right)=\left. \frac{\partial f(x + \eta p)}{\partial \eta} \right|_{\eta=0}.
# $$
# 
# Note that 
# 
# $$
#     \begin{split}
# \left. \frac{\partial f(x + \eta p)}{\partial \eta} \right|_{\eta=0}=\sum_{i=1}^n\left. \frac{\partial f}{\partial x_i}(x + \eta p)p_i \right|_{\eta=0} =(\nabla f, p),
# \end{split}
# $$
# 
# which means that
# 
# $$
#     f(x+\eta p)-f(x)=\eta(\nabla f(x), p)+o(\eta) .
# $$
# 
# Therefore
# 
# $$
#     \Delta(p)=(\nabla f(x), p).
# $$
# 
# Using the Cauchy-Schwarz inequality
# $-\|x\| \cdot\|y\| \leq( x, y) \leq\|x\| \cdot\|y\|,$ we obtain
# 
# $$
#     -\|\nabla f(x)\| \le (\nabla f(x), p)\le \|\nabla f(x)\| .
# $$
# 
# Let us take
# 
# $$
#     \bar{p}=-\nabla f(x) /\|\nabla f(x)\|.
# $$
# 
# Then
# 
# $$
#     \Delta(\bar{p})=-(\nabla f(x), \nabla f(x)) /\|\nabla f(x)\|=-\|\nabla f(x)\|.
# $$
# 
# The direction $-\nabla f(x)$ (the antigradient) is the direction of the
# fastest local decrease of the function $f(\cdot)$ at point $x.$ ◻
# ```
# 
# Here is a simple diagram for this property.
# 
# Since at each point, $f(x)$ decreases most rapidly along the negative
# gradient direction, it is then natural to choose the search direction in
# {eq}`line-search`  in the negative gradient direction and the
# resulting algorithm is the so-called gradient descent method.
# 
# ```{prf:algorithm} Algrihthm
# :label: my_algorithm1
# Given the initial guess $x_0$, learning rate $\eta_t>0$
# 
# **For** t=1,2,$\cdots$,
# 
# $$
#     x_{t+1} =  x_{t} - \eta_{t} \nabla f({x}_{t}),
# $$
# 
# ```
# 
# 
# 
# In practice, we need a "stopping criterion" that determines when the
# above gradient descent method to stop. One possibility is
# 
# > **While** $S(x_t; f) = \|\nabla f(x_t)\|\le \epsilon$ or $t \ge T$
# 
# for some small tolerance $\epsilon>0$ or maximal number of iterations
# $T$. In general, a good stopping criterion is hard to come by and it is
# a subject that has called a lot of research in optimization for machine
# learning.
# 
# In the gradient method, the scalar factors for the gradients,
# $\eta_{t},$ are called the step sizes. Of course, they must be positive.
# There are many variants of the gradient method, which differ one from
# another by the step-size strategy. Let us consider the most important
# examples.
# 
# 1.  The sequence $\left\{\eta_t\right\}_{t=0}^{\infty}$ is chosen in
#     advance. For example, (constant step)
#     
#     $$
#         \eta_t=\frac{\eta}{\sqrt{t+1}};
#     $$
# 
# 2.  Full relaxation:
# 
#     $$
#         \eta_t=\arg \min _{\eta \geq 0} f\left(x_t-\eta \nabla f\left(x_t\right)\right);
#     $$
# 
# 3.  The Armijo rule: Find $x_{t+1}=x_t-\eta \nabla f\left(x_t\right)$
#     with $\eta>0$ such that
#     
#     $$
#         \alpha\left(\nabla f\left(x_t\right), x_t-x_{t+1}\right) \leq f\left(x_t\right)-f\left(x_{t+1}\right),
#     $$
#     
#     $$
#         \beta\left(\nabla f\left(x_t\right), x_t-x_{t+1}\right) \geq f\left(x_t\right)-f\left(x_{t+1}\right),
#     $$
#     
#     where $0<\alpha<\beta<1$ are some fixed parameters.
# 
# Comparing these strategies, we see that
# 
# 1.  The first strategy is the simplest one. It is often used in the
#     context of convex optimization. In this framework, the behavior of
#     functions is much more predictable than in the general nonlinear
#     case.
# 
# 2.  The second strategy is completely theoretical. It is never used in
#     practice since even in one-dimensional case we cannot find the exact
#     minimum in finite time.
# 
# 3.  The third strategy is used in the majority of practical algorithms.
#     It has the following geometric interpretation. Let us fix
#     $x \in \mathbb{R}^{n}$ assuming that $\nabla f(x) \neq 0$. Consider
#     the following function of one variable:
#     
#     $$
#         \phi (\eta)=f(x-\eta \nabla f(x)),\quad \eta\ge0.
#     $$
#     
#     Then the
#     step-size values acceptable for this strategy belong to the part of
#     the graph of $\phi$ which is located between two linear functions:
#     
#     $$
#         \phi_{1}(\eta)=f(x)-\alpha \eta\|\nabla f(x)\|^{2}, \quad \phi_{2}(\eta)=f(x)-\beta \eta\|\nabla f(x)\|^{2}
#     $$
#     
#     Note that $\phi(0)=\phi_{1}(0)=\phi_{2}(0)$ and
#     $\phi^{\prime}(0)<\phi_{2}^{\prime}(0)<\phi_{1}^{\prime}(0)<0 .$
#     Therefore, the acceptable values exist unless $\phi(\cdot)$ is not
#     bounded below. There are several very fast one-dimensional
#     procedures for finding a point satisfying the Armijo conditions.
#     However, their detailed description is not important for us now.
#     
# 
# ##  Convergence of Gradient Descent method
# 
# Now we are ready to study the rate of convergence of unconstrained
# minimization schemes. For the optimization problem {eq}`problem`
# 
# 
# $$
#     \min_{x\in \mathbb{R}^n} f(x).
# $$
# 
# We assume that $f(x)$ is convex. Then we say that $x^*$ is a minimizer if
# 
# $$
#     f(x^*) = \min_{x \in \mathbb{R}^n} f(x).
# $$
# 
# For minimizer $x^*$, we have
# 
# $$
#     \label{key}
#     \nabla f(x^*) = 0.
# $$
# 
# We have the next two properties of the minimizer
# for convex functions:
# 
# 1.  If $f(x) \ge c_0$, for some $c_0 \in \mathbb{R}$, then we have
# 
#     $$
#         \mathop{\arg\min} f \neq \emptyset.
#     $$
# 
# 2.  If $f(x)$ is $\lambda$-strongly convex, then $f(x)$ has a unique
#     minimizer, namely, there exists a unique $x^*\in \mathbb{R}^n$ such
#     that
#     
#     $$
#         f(x^*) = \min_{x\in \mathbb{R}^n }f(x).
#     $$
# 
# To investigate the convergence of gradient descent method, let us recall
# the gradient descent method:
# 
# ```{prf:algorithm} Algorithm
# :label: my_algorithm2
# 
# **For**: $t = 1, 2, \cdots$ 
#  
# $$
#     \label{equ:fgd-iteration}
#     x_{t+1} =  x_{t} - \eta_t \nabla f(x_t),
# $$
# 
# where $\eta_t$ is the stepsize / learning rate.
# ```
# 
# We have the next theorem about the convergence of gradient descent
# method under the Assumption.
# 
# ```{admonition} Theorem
# For Gradient Descent Algorithm {prf:ref}`my_algorithm2` , if
# $f(x)$ satisfies Assumption, then
# 
# $$
#     \|x_t - x^*\|^2 \le  \alpha^t \|x_0 - x^*\|^2
# $$
# 
# if $0<\eta_t <\frac{2\lambda}{L^2}$ and $\alpha < 1$.
# 
# Particularly, if $\eta_t = \frac{\lambda}{L^2}$, then
# 
# $$
#     \|x_t - x^*\|^2 \le  \left(1 - \frac{\lambda^2}{L^2}\right)^t \|x_0 - x^*\|^2.
# $$
# ```
# 
# ```{admonition} Proof
# *Proof.* Note that 
# 
# $$
#     x_{t+1} - x =  x_{t} - \eta_t \nabla f(x_t)  - x.
# $$
# 
# By taking $L^2$ norm for both sides, we get
# 
# $$
#     \|x_{t+1} - x \|^2 = \|x_{t} - \eta_t \nabla f(x_t) - x \|^2.
# $$
# 
# Let
# $x = x^*$. It holds that 
# 
# $$
#     \begin{aligned}
#     \|x_{t+1} - x^* \|^2 &=  \| x_{t} - \eta_t \nabla f(x_t) - x^* \|^2 \\
#     &= \|x_t-x^*\|^2 - 2\eta_t \nabla f(x_t)^\top (x_t - x^*) + \eta_t^2 \|\nabla f(x_t) - \nabla f(x^*)\|^2 \qquad \mbox{ (by $\nabla f(x^*)=0$)}\\
#     &\le \|x_t - x^*\|^2 - 2\eta_t \lambda \|x_t - x^*\|^2 + \eta_t ^2 L^2 \|x_t - x^*\|^2  \quad
#     \mbox{(by $\lambda$- strongly convex \eqref{strongConvIneq} and Lipschitz)}\\
#     &\le (1 - 2\eta_t \lambda + \eta_t^2 L^2) \|x_t - x^*\|^2
#     =\alpha \|x_t - x^*\|^2,
#     \end{aligned}
# $$
# 
# where
# 
# $$
#     \alpha = \left(L^2 (\eta_t  -{\lambda\over L^2})^2 + 1-{\lambda^2\over L^2}\right)<1\  \mbox{if } 0< \eta_t<\frac{2\lambda}{L^2}.
# $$
# 
# Particularly, if $\eta_t =\frac{\lambda}{L^2}$,
# 
# $$
#     \alpha=1-{\lambda^2\over L^2},
# $$ 
# 
# which finishes the proof. ◻
# ```
# 
# This means that if the learning rate is chosen appropriatly,
# $\{x_t\}_{t=1}^\infty$ from the gradient descent method will converge to
# the minimizer $x^*$ of the function.
# 
# There are some issues on Gradient Descent method:
# 
# -   $\nabla f(x_{t})$ is very expensive to compute.
# 
# -   Gradient Descent method does not yield generalization accuracy.
# 
# The stochastic gradient descent (SGD) method in the next section will
# focus on these two issues.

# In[ ]:




