#!/usr/bin/env python
# coding: utf-8

# # Logistic regression 

# In[1]:


from IPython.display import IFrame
IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_3im0zbc7&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_ciutwbnv" ,width='800', height='500')


# In[2]:


IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_awemnq71&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_hjxe3ipe" ,width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2021/12/A03LogisticRegression_Video_Notes.pdf)

# Assume that we are given $k$ linearly separable sets
# $A_1,A_2,\cdots,A_k\in \mathbb{R}^d$, we define the set of classifiable
# weights as
# 
# $$
#     \mathbf\Theta = \{\mathbf\theta = (W,b): w_ix+b_i>w_jx+b_j,~\forall x\in A_i, j\neq i, i= 1,\cdots,k\}
# $$
# 
# which means those $(W,b)$ can separate $A_1,A_2,\cdots,A_k$ correctly.
# 
# Our linearly separable assumption implies that
# $\mathbf\Theta\neq \emptyset$. Now we know the existence of linearly
# classifiable weights. But how can we find one element in $\mathbf\Theta$?
# 
# ```{adnomition} Definition(softmax)
# :class: tip
# :name: softmax
# Given $s = (s_1,s_2,\cdots,s_k)^T\in \mathbb{R}^k$, we define the soft-max
# mapping $\sigma: \mathbb{R}^k \rightarrow\mathbb{R}^k$ as
# 
# $$
#     \sigma(s)  = \frac{e^{s}}{e^{s}\cdot \mathbf{1}} = \frac{1}{\sum\limits_{i=1}^k e^{s_i}}
#     \begin{pmatrix}
#     e^{s_1}\\
#     e^{s_2}\\
#     \vdots\\
#     e^{s_k}
#     \end{pmatrix}
# $$ 
# 
# where $e^{s} = 
# \begin{pmatrix}
# e^{s_1}\\
# e^{s_2}\\
# \vdots\\
# e^{s_k}
# \end{pmatrix}$, $\mathbf{1} = 
# \begin{pmatrix}
# 1\\
# 1 \\
# \vdots \\
# 1
# \end{pmatrix} \in\mathbb{R}^k$.
# ```
# 
# ```{admonition} Definition
# Given parameter $\mathbf\theta = (W,b)$, we define a feature mapping
# $\mathbf p: \mathbb{R}^d \rightarrow \mathbb{R}^k$ as
# 
# $$
#     \mathbf p(x; \mathbf\theta)  = \sigma(Wx+b) = \frac{1}{\sum\limits_{i=1}^k e^{w_i x+b_i}}
#     \begin{pmatrix}
#     e^{w_1 x+b_1}\\
#     e^{w_2 x+b_2}\\
#     \vdots\\
#     e^{w_k x+b_k}
#     \end{pmatrix}
#     = \begin{pmatrix}
#     p_1(x; \mathbf\theta) \\
#     p_2(x; \mathbf\theta) \\
#     \vdots \\
#     p_k(x; \mathbf\theta)
#     \end{pmatrix}
# $$
#     
# where the $i$-th component 
#     
# $$
#     p_i(x; \mathbf\theta) = \frac{e^{w_i x+b_i}}{\sum\limits_{i=1}^k e^{w_i x+b_i}}.
# $$ (key)
# ```
# 
# 
# The soft-max mapping have several important properties.
# 
# 1.  $\displaystyle 0< p_i(x; \mathbf\theta) <1,~\sum_i p_i(x; \mathbf\theta) = 1$.
# 
#     This implies that $\mathbf p(x; \mathbf\theta)$ can be regarded as a
#     probability distribution of data points which means that given
#     $x\in \mathbb{R}^d$, we have $x\in A_i$ with probability
#     $p_i(x; \mathbf{\theta})$, $i = 1,\cdots,k$.
# 
# 2.  $p_i(x; \mathbf\theta)>p_j(x; \mathbf\theta)\Leftrightarrow w_ix+b_i>w_j x+b_j.$
# 
#     This implies that the linearly classifiable weights have an
#     equivalent description as
#     
# $$
#     \mathbf{\Theta} = \left\{\mathbf\theta: p_i(x; \mathbf\theta)>p_j(x,\mathbf\theta),~\forall x\in A_i, j\neq i, i= 1,\cdots,k\right\}
# $$
# 
# 3.  We usually use the max-out method to do classification. For a given
#     data point $x$, we first use a soft-max mapping to map it to
#     $\mathbf p(x; \mathbf\theta)$, then we attach $x$ to the class
#     $i= \arg\max_j p_i(x; \mathbf\theta)$.
# 
#     This means that we pick the label $i$ as the class of $x$ such that
#     $x\in A_i$ has the biggest probability $p_i(x; \mathbf\theta)$.
# 
# more detailed discussion of logistic regression from the probability
# perspective will be presented in the nearly future.
# 
# From the above properties, we can define the following likelihood
# function to help find elements in $\mathbf{\Theta}$: 
# 
# $$
# P (\mathbf\theta)=
# \prod\limits_{i = 1}^k \prod\limits_{x\in A_i} p_i(x; \mathbf\theta).
# $$
#     
# Based on the property that 
# 
# $$
# p_i (x; \mathbf \theta) = \max_{1\le j \le k} p_j(x; \mathbf \theta), \quad\forall x \in A_i,\ \mathbf \theta \in \Theta,
# $$ (key1)
# 
# we may use the next optimization problem 
# 
# $$ 
# \max_{\mathbf \theta\in \mathbf{\Theta}} P(\mathbf \theta).
# $$ (key2)
#     
# to find an element in
# $\mathbf{\Theta}$. more precisely, let us introduce the next lemmas
# (properties) of $P(\mathbf \theta)$.
# 
# ```{admonition} Lemma
# :name: lemmH12
# Assume that the sets
# $A_1,A_2,\cdots,A_k$ are linearly separable. Then we have
# 
# $$
#     \left\{\mathbf \theta:~P(\mathbf\theta)>\frac{1}{2}\right\}\subset \mathbf{\Theta}.
# $$
# ```
# 
# ```{admonition} Proof
# *Proof.* It suffices to show that if $\mathbf\theta \not\in \mathbf\Theta$, we
# must have $P(\mathbf\theta)\leq\frac{1}{2}$. For any $\mathbf\theta \not\in
#     \mathbf\Theta$, there must exist an $i_0$ ,an $x_0\in A_{i_0}$ and a
# $j_0\neq i_0$ such that
# 
# $$
#     w_{i_0} x_0 + b_{i_0} \leq w_{j_0}x_0 + b_{j_0}.
# $$ 
# 
# Then we have
# 
# $$
#     p_{i_0}(x_0; \mathbf\theta) \leq \frac{e^{w_{i_0} x_0 + b_{i_0}}}{e^{w_{i_0} x_0+b_{i_0}}+e^{w_{j_0} x_0+b_{j_0}}} \leq\frac{1}{2}.
# $$
#     
# Notice that $p_i(x; \mathbf \theta) < 1$ for all $i = 1,\cdots,k$, $x\in A$.
# So 
# 
# $$
#     P(\mathbf\theta) <  p_{i_0}(x_0; \mathbf\theta) \leq \frac{1}{2}.
# $$ 
# ```
# 
# ```{admonition} Lemma
# If $A_1,A_2,\cdots,A_k$ are linearly separable and
# $\mathbf\theta \in \mathbf\Theta$, we have
# 
# $$
#     \lim_{\alpha\rightarrow +\infty}p_i(x; \alpha\mathbf\theta) = 1\Leftrightarrow x\in A_i.
# $$
# ```
# 
# ```{admonition} Proof
# *Proof.* We first note that if $x\in A_i$
# 
# $$
#     p_i(x,\mathbf \theta) = \frac{1}{1+\sum\limits_{j\neq i}e^{\alpha[(w_j x+ b_j)-(w_i x+b_i)]}} \to 1, \quad \text{as} \quad \alpha \to \infty.
# 
# $$
#     
# On the other hand, if $x\not\in A_i$,
# 
# $$
#     p_i(x; \mathbf\alpha\mathbf\theta) = \frac{1}{1+\sum\limits_{j\neq i}e^{\alpha[(w_j x+ b_j)-(w_i x+b_i)]}} \leq \frac{1}{2}.
# $$
#     
# This implies that if $x\not\in A_i$,
# $\lim_{\alpha\rightarrow \infty}p_i(x; \alpha\mathbf \theta)\neq 1$ which is
# equivalent to the proposition that if
# $\lim_{\alpha\rightarrow \infty}p_i(x; \alpha\mathbf \theta)= 1$, then
# $x\in A_i$. 
# ```
# 
# ```{admonition} Lemma
# :name: thm1
# If $A_1,A_2,\cdots,A_k$ are linearly
# separable,
# 
# $$
#     \mathbf\Theta = \left\{\mathbf\theta: \lim_{\alpha\rightarrow +\infty}P(\alpha\mathbf\theta) = 1\right\}.
# $$
# ```
# 
# ```{admonition} Proof
# *Proof.* We first note that if $\mathbf\theta \in\mathbf\Theta$, we have 
# $\displaystyle\lim_{\alpha\rightarrow +\infty}p_i(x; \alpha\mathbf\theta) = 1$
# for all $x\in A_i$. So
# 
# $$
#     \lim\limits_{\alpha\rightarrow +\infty} P(\alpha\mathbf\theta) = \lim\limits_{\alpha\rightarrow +\infty} \prod\limits_{i = 1}^k \prod\limits_{x\in A_i} p_i(x; \alpha\mathbf\theta) = \prod\limits_{i = 1}^k \prod\limits_{x\in A_i} \lim\limits_{\alpha\rightarrow +\infty}p_i(x; \alpha\mathbf\theta) = 1.    
# $$
#     
# On the other hand, if
# $\lim\limits_{\alpha\rightarrow +\infty} P(\alpha\mathbf\theta) = 1$, there
# must exist one $\alpha_0>0$ such that
# $P(\alpha_0\mathbf\theta) >\frac{1}{2}$. From [Lemma](lemmH12), we have $\alpha_0\mathbf\theta\in\mathbf\Theta$, which
# means $\mathbf\theta\in\mathbf\Theta$. 
# ```
# 
# These properties above imply that if we can obtain a classifiable weight
# through maximizing $P(\mathbf\theta)$, while [Lemma](thm1) tells us that
# $P(\mathbf\theta)$ will not have a global minimum actually.
# 
# more specifically, we just need to find some $\mathbf \theta \in \Theta$
# such that 
# 
# $$
# P(\mathbf \theta) > \frac{1}{2} \Leftrightarrow  L(\mathbf \theta) : = -\log P(\mathbf \theta )  < \log(2).
# $$
# 
# ## Regularized logistic regression
# 
# Here, we start from the regularization term
# $e^{-\lambda R(\|\mathbf\theta\|)}$ with these next properties:
# 
# 1.  $\lambda > 0$.
# 
# 2.  $R(t)$ is a strictly increasing function on $\mathbb{R}^+$ with
#     $R(0) = 0$, $\lim\limits_{t\rightarrow +\infty} R(t) = +\infty$. For
#     example, $R(t) = t^2$.
# 
# 3.  $\|\cdot\|$ is a norm on $R^{k\times(d+1)}$, a commonly used norm is
#     the following Frobenius norm: $$\label{key}
#         \|\mathbf \theta\|_F = \sqrt{\sum_{i,j}W_{ij}^2 + \sum_i b_i^2}.$$
# 
# Based on this regularization term, we may consider the following
# regularized likelihood function $P_\lambda(\mathbf\theta)$ as 
# 
# $$
#     P_\lambda(\mathbf\theta) = P(\mathbf\theta)e^{-\lambda R(\|\mathbf\theta\|)}.
# $$
# 
# Here, let us define 
# 
# $$
#     \mathbf\Theta_{\lambda} = \mathop{{\arg\max}}_{\mathbf\theta}  P_\lambda(\mathbf\theta),
# $$
# 
# where 
# 
# $$
#     \mathop{\arg\max}_{\mathbf\theta}  P_\lambda(\mathbf\theta) = \left\{\mathbf \theta ~:~ P_\lambda(\mathbf \theta) = \max_{\mathbf \theta} P_\lambda(\mathbf \theta) \right\}.
# $$
# 
# The next lemma show that the maximal set of modified objective is not
# empty.
# 
# ```{admonition} Lemma
# Suppose that $A_1,A_2, \cdots, A_k$ are linearly separable, then
# 
# 1.  if $\lambda = 0$, $\mathbf\Theta_0 = \emptyset$,
# 
# 2.  $\mathbf\Theta_{\lambda}$ must be nonempty for all $\lambda>0$.
# ```
# 
# ```{admonition} Proof
# *Proof.* [Lemma](thm2)
# shows the first proposition. For the second proposition, we notice that
# 
# 1.  $P_\lambda(\mathbf 0) = \frac{1}{k^N}$.
# 
# 2.  $\exists\ m_{\lambda}>0$ such that
#     $e^{-\lambda R(\|\mathbf\theta\|)}<\frac{1}{k^N}$ whenever
#     $\|\mathbf\theta\|> m_{\lambda}$ because of the properties of
#     $R(\|\mathbf\theta\|)$.
# 
# So a maxima on $\{\mathbf\theta: \|\mathbf\theta\| \le m_{\lambda}\}$ must be a
# global maxima. Then we can easily obtain the result in the lemma from
# the boundedness and closeness of
# $\{\mathbf\theta: \|\mathbf\theta\| \le m_{\lambda}\}$. 
# ```
# 
# Furthermore, we have the next theorem which shows that we can indeed get
# $\Theta$ by maximizing $P_\lambda(\mathbf \theta)$.
# 
# ```{admonition} Theorem
# :name: thm-L-Theta
# If $A_1,A_2,\cdots,A_k$ are linearly separable, 
# 
# $$
#     \mathbf\Theta_{\lambda} \subset \mathbf\Theta,
# $$
# 
# when $\lambda>0$ and sufficiently small.
# ```
# 
# ```{admonition} Proof
# *Proof.* By [Lemma](lemmH12)  we can take $\mathbf\theta_0\in \mathbf\Theta$ such that
# $P(\mathbf\theta_0)> \frac{3}{4}$. Then, for any
# $\lambda < \frac{\log \frac{3}{2}}{R(\|\mathbf\theta_0\|)}$,
# $\mathbf\theta_{\lambda}\in \mathbf\Theta_{\lambda}$, we have
# 
# $$
#     P(\mathbf\theta_{\lambda}) \geq  P_\lambda(\mathbf\theta_{\lambda})  \geq P_\lambda(\mathbf \theta_0) = P(\mathbf\theta_0)e^{-\lambda R(\|\mathbf\theta_0\|)} > \frac{3}{4}\cdot \frac{2}{3} = \frac{1}{2},
# $$
# 
# which implies that $\mathbf \theta_{\lambda} \in \Theta$. Thus, for any
# $0< \lambda < \frac{\log \frac{3}{2}}{R(\|\mathbf\theta_0\|)}$,
# $\mathbf\Theta_{\lambda} \subset \mathbf\Theta$. ◻
# ```
# 
# The design of logistic regression is that maximize
# $P_\lambda(\mathbf\theta)$ is equivalent to minimize
# $-\log P_\lambda(\mathbf\theta)$, i.e., 
# 
# $$
#     \max_{\mathbf \theta} \left\{ P_\lambda(\mathbf\theta) \right\} \Leftrightarrow \min_{\mathbf \theta} \left\{ -\log   P_\lambda(\mathbf\theta)\right\},
# $$
# 
# while the second one is more convenient to evaluate the gradient.
# meanwhile, we add a regularization term $R(\mathbf\theta)$ to the objective
# function which makes the optimization problem has a unique solution.
# 
# mathematically, we can formulate Logistic regression as
# 
# $$
#     \min_{\mathbf\theta} L_\lambda(\mathbf \theta),
# $$ 
# 
# where
# 
# $$
#     L_\lambda(\mathbf \theta)  := -\log P_\lambda(\mathbf\theta) = -\log P(\mathbf\theta) + \lambda R(\|\mathbf\theta\|) = L(\mathbf\theta) + \lambda R(\|\mathbf\theta\|),
# $$(logisticlambda)
# 
# with 
# 
# $$
# L(\mathbf \theta) = - \sum_{i=1}^k \sum_{x\in A_i} \log p_{i}(x;\mathbf \theta).
# $$(logistic)
# 
# Then we have the next logistic regression algorithm.
# 
# ```{admonition} Algorithm
# Given data $A_1, A_2, \cdots, A_k$, find 
# 
# $$
#     \mathbf \theta^* = \mathop{\arg\min}_{\mathbf \theta}  L_\lambda(\mathbf \theta),
# $$
# 
# for some sufficient small $\lambda > 0$.
# ```
# 
# ```{admonition} Remark
# Here 
# 
# $$
#     L(\mathbf \theta)  = -\log P(\mathbf\theta),
# $$ 
# 
# is known as the loss function of logistic regression model. The next reasons may show that why $L(\mathbf \theta)$ is popular.
# 
# 1.  It is more convenient to take gradient for $L(\mathbf \theta)$ than
#     $P(\mathbf \theta)$.
# 
# 2.  $L(\mathbf \theta)$ is related the so-called cross-entropy loss function
#     which will be discussed in the next section.
# 
# 3.  $L(\mathbf \theta)$ is a convex function which will be discussed later.
# ```

# In[ ]:




