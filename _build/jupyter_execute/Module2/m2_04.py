#!/usr/bin/env python
# coding: utf-8

# # Stochastic gradient descent method and convergence theory

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_80cz6xki&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_ca2yxlxv" ,width='800', height='500')


# In[2]:


IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_9g3lrg1l&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_ddb7qq2t",width='800', height='500') 


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2022/01/B04SGDConvergence_Video_Notes.pdf)

# ## Stochastic gradient descent method and convergence theory
# 
# The next optimization problem is the most common case in machine
# learning.
# 
# ```{admonition} Problem
# :label: SGDproblem
# 
# $$
#     \min_{x \in \mathbb{R}^n} f(x)\quad \mbox{and}\quad f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x).
# $$
# 
# ```
# 
# One version of stochastic gradient descent (SGD) algorithm is:
# 
# ```{prf:algorithm} SGD
# **Input**: initialization $x_0$, learning rate $\eta_t$.
# 
# **For**: t = 0,1,2,$\dots$
# 
# Randomly pick $i_t \in \{1, 2, \cdots, N\}$ independently with
# probability $\frac{1}{N}$ 
# 
# $$
#     x_{t+1} = x_{t} - \eta_t \nabla f_{i_t}(x_t).
# $$
# 
# ```
# 
# ### Convergence of SGD
# 
# ```{prf:theorem} 
# Assume that each $f_i(x)$ is $\lambda$-strongly convex and
# $\|\nabla f_i(x)\| \le M$ for some $M >0$. If we take
# $\eta_t = \frac{a}{\lambda (t+1)}$ with sufficiently large $a$ such that
# 
# $$
#     \|x_0 - x^*\|^2 \le \frac{a^2M^2}{(a-1)\lambda^2}
# $$
# 
# then
# 
# $$
#     \mathbb{E}e_{t}^2 \le \frac{a^2M^2}{(a-1)\lambda^2 (t+1)}, \quad  t\ge 1,
# $$
# 
# where $e_t = \|x_t - x^*\|$.
# ```
# 
# ```{prf:proof} Proof
# *Proof.* The $L^2$ error of SGD can be written as 
# 
# $$
#       \begin{split}
#             \mathbb{E} \|x_{t+1} - x^*\|^2 &\le \mathbb{E}\| x_{t} - \eta_t \nabla f_{i_t}(x_t) - x^* \|^2 \\
#             &\le \mathbb{E} \|x_t - x^*\|^2 
#             - 2 \eta_t \mathbb{E} (\nabla f_{i_t}(x_t) \cdot (x_t - x^*)) 
#             + \eta_t^2 \mathbb{E} \|\nabla f_{i_t}(x_t)\|^2 \\
#             & \le \mathbb{E} \|x_t - x^*\|^2 - 2 \eta_t \mathbb{E} (\nabla f (x_t) \cdot (x_t - x^*))
#             + \eta_t^2 M^2 \\
#             & \le \mathbb{E} \|x_t - x^*\|^2 -  \eta_t \lambda \mathbb{E} \|x_t - x^*\|^2 + \eta_t^2 M^2 \\
#             & = (1 - \eta_t\lambda) \mathbb{E} \|x_t - x^*\|^2 + \eta_t^2 M^2
#       \end{split}
# $$
# 
# The third line comes from the fact that
# 
# $$
#     \begin{aligned}
#     \mathbb{E} (\nabla f_{i_t}(x_t) \cdot (x_t - x^*))  &= \mathbb{E}_{i_1i_2\cdots i_t} (\nabla f_{i_t}(x_t) \cdot (x_t - x^*)) \\
# &= \mathbb{E}_{i_1i_2\cdots i_{t-1}} \frac{1}{N} \sum_{i=1}^N \nabla f_i(x_t)\cdot (x_t - x^*) \\
# &= \mathbb{E}_{i_1i_2\cdots i_{t-1}}  \nabla f(x_t)\cdot (x_t - x^*) \\
# &= \mathbb{E}\nabla f(x_t)\cdot (x_t - x^*),
# \end{aligned}
# $$ 
# 
# and
# 
# $$
#     \mathbb{E} \|\nabla f_{i_t}(x_t)\|^2 \le \mathbb{E} M^2 = M^2.
# $$
# 
# Note when $t=0$, we have 
# 
# $$
#     \mathbb{E} e_0^2 = \|x_0 - x^*\|^2 \le \frac{a^2M^2}{(a-1)\lambda},
# $$
# 
# based on the assumption.
# 
# In the case of SDG, by the inductive hypothesis, 
# 
# $$
#     \begin{split}
#             \mathbb{E}e_{t+1}^2 & \le (1 - \eta_t\lambda)\mathbb{E}e_{t}^2  + \eta_t^2 M^2\\
#             &\le  (1 - \frac{a}{t+1}) \frac{a^2M^2}{(a-1)\lambda^2 (t+1)} + \frac{a^2M^2}{\lambda^2 (t+1)^2} \\
#             & \le \frac{a^2M^2}{(a-1)\lambda^2} \frac{1}{(t+1)^2}(t+1 -a + a-1) \\
#             & = \frac{a^2M^2}{(a-1)\lambda^2} \frac{t}{(t+1)^2} \\
#             & \le \frac{a^2M^2}{(a-1)\lambda^2(t+2)}. \quad \left(\frac{t}{(t+1)^2} \le \frac{1}{t+2}\right),
#       \end{split}
# $$
# 
# which completes the proof. ◻
# ```
# 
# ### SGD with mini-batch
# 
# Firstly, we will introduce a natural extended version of the SGD
# discussed above with introducing mini-batch.
# 
# ```{prf:algorithm} SGD with mini-batch
# **Input**: initialization $x_0$, learning rate $\eta_t$.
# 
# **For**: t = 0,1,2,$\dots$
# 
# 
# Randomly pick $B_t \subset \{1, 2, \cdots, N\}$ independently with
# probability $\frac{m!(N-m)!}{N!}$\
# and $\# B_t = m$.
# 
# $$
#     x_{t+1} = x_{t} - \eta_t g_t(x_t).
# $$
# 
# where
# 
# $$
#     g_{t}(x_t) = \frac{1}{m} \sum_{i \in B_{t}}  \nabla f_i(x_t)
# $$
# 
# ```
# 
# Now we introduce the SGD algorithm with mini-batch without replacement
# which is the most commonly used version of SGD in machine learning.
# 
# ```{prf:algorithm} Shuffle SGD with mini-batch
# **Input**: learning rate $\eta_k$, mini-batch size $m$, parameter
# initialization $x_{0}$ and denote $M = \lceil \frac{N}{m} \rceil$.
# 
# **For** Epoch $k = 1,2,\dots$
# 
# 
# Randomly pick $B_t \subset \{1, 2, \cdots, N \}$ without replacement\
# with $\# B_t = m$ for $t = 1,2,\cdots,M$.
# 
# 
# mini-batch $t = 1:M$
# 
# Compute the gradient on $B_{t}$:
#     
# $$
#     g_{t}(x) = \frac{1}{m} \sum_{i \in B_{t}}  \nabla f_i(x)
# $$
# 
# Update $x$:
# 
# $$
#     x  \leftarrow  x - \eta_k g_t(x),
# $$
# 
# **EndFor**
# ```
# 
# To \"randomly pick $B_i \subset \{1, 2, \cdots, N \}$ without
# replacement with $\# B_i = m$ for $i = 1,2,\cdots,t$", we usually just
# randomly shuffle the index set first and then consecutively pick every
# $m$ elements in the shuffled index set. That is the reason why we would
# like to call the algorithm as shuffled SGD while this is the mostly used
# version of SGD in machine learning.
# 
# ```{admonition} Remark
# Let us recall a general machine learning loss function 
# 
# $$
#     \label{key}
#     L(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(h(X_i; \theta), Y_i),
# $$
# 
# where
# $\{(X_i, Y_i)\}_{i=1}^N$ correspond to these data pairs. For example,
# $\ell(\cdot, \cdot)$ takes cross-entropy and
# $h(x; \theta) = p(x;\theta)$ as we discussed in Section 2.2.1. Thus, we
# have the following corresponding relation
# 
# $$
#     f(x) \leftrightsquigarrow L(\theta), \quad
#     f_i(x) \leftrightsquigarrow \ell(h(X_i; \theta), Y_i).
# $$
#     
# ```
# 

# In[ ]:




