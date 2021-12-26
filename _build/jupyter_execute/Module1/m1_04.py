#!/usr/bin/env python
# coding: utf-8

# # KL-divergence and cross-entropy

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_1x5pta90&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_l1sjg1vv"  ,width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2021/12/A04CrossEntropy_Video_Notes.pdf)

# Cross-entropy minimization is frequently used in optimization and
# rare-event probability estimation. When comparing a distribution against
# a fixed reference distribution, cross-entropy and KL divergence are
# identical up to an additive constant. See more details in
# [@murphy2012machine; @kullback1951information; @kullback1997information]
# and the reference therein.
# 
# The KL(Kullback--Leibler) divergence defines a special distance between
# two discrete probability distributions 
# $
# p=\left( \begin{array}{ccc}
# p_1\\
# \vdots \\
# p_k
# \end{array} \right),\quad  q=\left( \begin{array}{ccc}
# q_1\\
# \vdots \\
# q_k
# \end{array} \right
# )
# $
# with $
# 0\le p_i, q_i\le1$
# and
# $\sum_{i=1}^{k}p_i=\sum_{i=1}^{k}q_i=1$ by $
# D_{\rm KL}(q,p)= \sum_{i=1}^k q_i\log \frac{q_i}{p_i}.$
# 
# ```{admonition} Lemma
# $D_{\rm KL}(q,p)$ works like a "distance\" without the symmetry:
# 
# 1.  $D_{\rm KL}(q,p)\ge0$;
# 
# 2.  $D_{\rm KL}(q,p)=0$ if and only if $p=q$;
# ```
# 
# ```{admonition} Proof
# *Proof.* We first note that the elementary inequality
# $\log x \le x - 1, \quad\mathrm{for\ any\ }x\ge0,$ and the equality
# holds if and only if $x=1$.
# $-D_{\rm KL}(q,p) = - \sum_{i=1}^c q_i\log \frac{q_i}{p_i}   = \sum_{i=1}^k q_i\log \frac{p_i}{q_i} \le \sum_{i=1}^k q_i( \frac{p_i}{q_i}  - 1) = 0.$
# And the equality holds if and only if
# $\frac{p_i}{q_i} = 1 \quad \forall i = 1:k.$ 
# ```
# 
# Define cross-entropy for distribution $p$ and $q$ by
# $
# H(q,p) = - \sum_{i=1}^k q_i \log p_i,$ and the entropy for distribution
# $q$ by $
# H(q) = - \sum_{i=1}^k q_i \log q_i.$ Note that
# $D_{\rm KL}(q,p)= \sum_{i=1}^k q_i\log \frac{q_i}{p_i} =  \sum_{i=1}^k q_i \log q_i - \sum_{i=1}^k q_i \log p_i$
# Thus, 
# 
# $$
#     H(q,p) = H(q) + D_{\rm KL}(q,p).
# $$ (rel1)
# 
# It follows from the [relation](rel1) that 
# 
# $$
#     \mathop{\arg\min}_p D_{\rm KL}(q,p)=\mathop{\arg\min}_p H(q,p).
# $$ (rel2)
# 
# The concept of cross-entropy can be used to define a loss function in
# machine learning and optimization. Let us assume $y_i$ is the true label
# for $x_i$, for example $y_i = e_{k_i}$ if $x_i \in A_{k_i}$. Consider
# the predicted distribution 
# $p(x;\theta) = \frac{1}{\sum\limits_{i=1}^k e^{w_i x+b_i}}$.
# $\begin{pmatrix}
# e^{w_1 x+b_1}\\
# e^{w_2 x+b_2}\\
# \vdots\\
# e^{w_k x+b_k}
# \end{pmatrix}
# = \begin{pmatrix}
# p_1(x; \theta) \\
# p_2(x; \theta) \\
# \vdots \\
# p_k(x; \theta)
# \end{pmatrix}$
# for any data $x \in A$. By[.](rel2), the minimization of KL divergence is
# equivalent to the minimization of the cross-entropy, namely
# $\mathop{\arg\min}_{\theta} \sum_{i=1}^N D_{\rm KL}(y_i, p(x_i;\theta)) = \mathop{\arg\min}_{\theta} \sum_{i=1}^N H(y_i,  p(x_i;  \theta)).$
# Recall that we have all data
# $D = \{(x_1,y_1),(x_2,y_2),\cdots, (x_N, y_N)\}$. Then, it is natural to
# consider the loss function as following:
# $\sum_{j=1}^N H(y_i,  p(x_i;  \theta)),$ which measures the
# distance between the real label and predicted one for all data. In the
# meantime, we can check that 
# $\begin{aligned}
# \sum_{j=1}^N H(y_j,  p(x_j;  \theta))&=-\sum_{j=1}^N y_j  \cdot \log   p(x_j;  \theta )\\
# &=-\sum_{j=1}^N  \log p_{i_j}(x_i; \theta) \quad (\text{because}~y_j = e_{i_j}~\text{for}~x_j \in A_{i_j})\\
# &=-\sum_{i=1}^k \sum_{x\in A_i}  \log p_{i}(x;  \theta) \\
# &=-\log \prod_{i=1}^k \prod_{x\in A_i}   p_{i}(x;  \theta)\\
# & = L(\theta)
# \end{aligned}$ with $L(\theta)$ 
# defined in as
# $L( \theta) = - \sum_{i=1}^k \sum_{x\in A_i} \log p_{i}(x; \theta).$
# 
# That is to say, the logistic regression loss function defined by
# likelihood in []() is exact the loss function defined by measuring
# the distance between real label and predicted one via cross-entropy. We
# can note $\label{key}
# \min_{ \theta} L_\lambda( \theta) \Leftrightarrow \min_{ \theta} \sum_{j=1}^N H(y_i,  p(x_i;  \theta)) + \lambda R(\| \theta\|) 
# \Leftrightarrow \min_{ \theta} \sum_{j=1}^N D_{\rm KL}(y_i, p(x_i;  \theta)) + \lambda R(\| \theta\|).$

# In[ ]:




