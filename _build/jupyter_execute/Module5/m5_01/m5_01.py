#!/usr/bin/env python
# coding: utf-8

# # Data normalization and weights initialization

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_01dyptyu&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_ri00enil"  ,width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2022/03/E01-02InitializationNormalization.pdf)

# ## Data normalization in DNNs and CNNs
# 
# ### Normalization for input data of DNNs
# 
# Consider that we have the all training data as
# 
# $$
#     (X, Y):=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{N}
# $$
# 
# for $x_{i} \in \mathbb{R}^{d}$ and $y_{i} \in \mathbb{R}^{k}$.
# 
# Before we input every data into a DNN model, we will apply the following
# normalization for all data $x_{i}$ for each component. Let denote
# 
# $$
#     \left[x_{i}\right]_{j} \longleftrightarrow \text { the } \mathrm{j} \text {-th component of data } x_{i}
# $$
# 
# Then we have following formula of for all $j=1,2, \cdots, d$
# 
# $$
#     \left[\tilde{x}_{i}\right]_{j}=\frac{\left[x_{i}\right]_{j}-\left[\mu_{X}\right]_{j}}{\sqrt{\left[\sigma_{X}\right]_{j}}}
# $$
# 
# where
# 
# $$
#     \left[\mu_{X}\right]_{j}=\mathbb{E}_{x \sim X}\left[[x]_{j}\right]=\frac{1}{N} \sum_{i=1}^{N}\left[x_{i}\right]_{j}, \quad\left[\sigma_{X}\right]_{j}=\mathbb{V}_{x \sim X}\left[[x]_{j}\right]=\frac{1}{N} \sum_{i=1}^{N}\left(\left[x_{i}\right]_{j}-\left[\mu_{X}\right]_{j}\right)^{2} 
# $$
# 
# Here $x \sim X$ means that $x$ is a discrete random variable on $X$ with
# probability 
# 
# $$
#     \mathbb{P}\left(x=x_{i}\right)=\frac{1}{N}
# $$
# 
# for any $x_{i} \in X$.
# 
# For simplicity, we rewrite the element-wise definition above as the
# following compact form
# 
# $$
#     \tilde{x}_{i}=\frac{x_{i}-\mu_{X}}{\sqrt{\sigma_{X}}}
# $$ 
# 
# where
# 
# $$
#     x_{i}, \tilde{x}_{i}, \mu_{X}, \sigma_{X} \in \mathbb{R}^{d}
# $$
# 
# defined as before and all operations in (1.6) are element-wise.
# 
# Here we note that, by normalizing the data set, we have the next
# properties for new data $\tilde{x} \in \tilde{X}$ with component
# $j=1,2, \cdots, d$,
# 
# $$
#     \mathbb{E}_{\bar{X}}\left[[\tilde{x}]_{j}\right]=\frac{1}{N} \sum_{i=1}^{N}\left[\tilde{x}_{i}\right]_{j}=0
# $$
# 
# and
# 
# $$
#     \mathbb{V}_{\tilde{X}}\left[[\tilde{x}]_{j}\right]=\frac{1}{N} \sum_{i=1}^{N}\left(\left[\tilde{x}_{i}\right]_{j}-\mathbb{E}_{\tilde{X}}\left[[\tilde{x}]_{j}\right]\right)^{2}=1
# $$
# 
# Finally, we will have a “new” data set
# 
# $$
#     \tilde{X}=\left\{\tilde{x}_{1}, \tilde{x}_{2}, \cdots, \tilde{x}_{N}\right\}
# $$
# 
# with unchanged label set $Y$. For the next sections, without special notices, we use $X$ data set as the normalized one as default.
# 
# ### Data normalization for images in CNNs
# 
# For images, consider we have a color image data set
# $(X, Y):=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{N}$ where
# 
# $$
#     x_{i} \in \mathbb{R}^{3 \times m \times n}
# $$
# 
# We further denote these the $(s, t)$ pixel value for data $x_{i}$ at channel $j$ as:
# 
# $$
#     \left[x_{i}\right]_{j ; s t} \longleftrightarrow(s, t) \text { pixel value for } x_{i} \text { at channel } j
# $$
# 
# where $1 \leq i \leq N, 1 \leq j \leq 3,1 \leq s \leq m$, and
# $1 \leq j \leq n .$
# 
# Then, the normalization for $x_{i}$ is defined by
# 
# $$
#     \left[\tilde{x}_{i}\right]_{j ; s t}=\frac{\left[x_{i}\right]_{j ; s t}-\left[\mu_{X}\right]_{j}}{\sqrt{\left[\sigma_{X}\right]_{j}}}
# $$
# 
# where
# 
# $$
#     \left[x_{i}\right]_{j ; s t},\left[\tilde{x}_{i}\right]_{j ; s t},\left[\mu_{X}\right]_{j},\left[\sigma_{X}\right]_{j} \in \mathbb{R}
# $$
# 
# Here
# 
# $$
#     \left[\mu_{X}\right]_{j}=\frac{1}{m \times n \times N} \sum_{1 \leq i \leq N} \sum_{1 \leq s \leq m, 1 \leq t \leq n}\left[x_{i}\right]_{j ; s t}
# $$
# 
# and
# 
# $$
#     \left[\sigma_{X}\right]_{j}=\frac{1}{N \times m \times n} \sum_{1 \leq i \leq N} \sum_{1 \leq s \leq m, 1 \leq t \leq n}\left(\left[x_{i}\right]_{j ; s t}-\left[\mu_{X}\right]_{j}\right)^{2}
# $$
# 
# In batch normalization, we confirmed with Lian by both numerical test
# and code checking that $\mathrm{BN}$ also use the above formula to
# compute the variance in $\mathrm{CNN}$ for each channel.
# 
# Another way to compute the variance over each channel is to compute the
# standard deviation on each channel for every data, and then average them
# in the data direction.
# 
# $$
#     \sqrt{\left[\tilde{\sigma}_{X}\right]_{j}}=\frac{1}{N} \sum_{1 \leq i \leq N}\left(\frac{1}{m \times n} \sum_{1 \leq s \leq m, 1 \leq t \leq n}\left(\left[x_{i}\right]_{j ; s t}-\left[\mu_{i}\right]_{j}\right)^{2}\right)^{\frac{1}{2}}
# $$
# 
# where
# 
# $$
#     \left[\mu_{i}\right]_{j}=\frac{1}{m \times n} \sum_{1 \leq s \leq m, 1 \leq t \leq n}\left[x_{i}\right]_{j ; s t} $$
# 
# ### Comparison of $\sqrt{\left[\sigma_{X}\right]_{j}}$ and $\sqrt{\left[\tilde{\sigma}_{X}\right]_{j}}$ on CIFAR10.
# 
# They share the same $\mu_{X}$ as $$\mu_{X}=\left(\begin{array}{lll}
# 0.49140105 & 0.48215663 & 0.44653168
# \end{array}\right)$$ But they had different standard deviation
# estimates: $$\begin{aligned}
# &\sqrt{\left[\sigma_{X}\right]_{j}}=(0.247032840 .243484990 .26158834) \\
# &\sqrt{\left[\tilde{\sigma}_{X}\right]_{j}}=(0.202201930 .199316350 .20086373)
# \end{aligned}$$
# 
# ##Initialization for deep neural networks
# 
# 
# ### Xavier’s Initialization
# 
# The goal of Xavier initialization \[1\] is to initialize the deep neural
# network to avoid gradient vanishing or blowup when the input is white
# noise.
# 
# Let us denote the DNN models as:
# 
# $$
#     \begin{cases}f^{1}(x) & =W^{1} x+b^{1} \\ f^{\ell}(x) & =W^{\ell} \sigma\left(f^{\ell-1}(x)\right)+b^{\ell} \quad \ell=2: L, \\ f(x) & =f^{L}\end{cases}
# $$
# 
# with $x \in \mathbb{R}^{n_{0}}$ and
# $f^{\ell} \in \mathbb{R}^{n_{\ell}}$. More precisely, we have
# 
# $$
#     W^{\ell} \in \mathbb{R}^{n_{\ell} \times n_{\ell-1}} 
# $$ 
# 
# The basic assumptions that we make are:
# 
# -   The initial weights $W_{i j}^{\ell}$ are i.i.d symmetric random
#     variables with mean 0, namely the probability density function of
#     $W_{i j}^{\ell}$ is even.
# 
# -   The initial bias $b^{\ell}=0$.
# 
# Now we choose the variance of the initial weights to ensure that the
# features $f^{L}$ and gradients don’t blow up or vanish. To this end we
# have the following lemma.
# 
# Lemma 1. Under the previous assumptions $f_{i}^{\ell}$ is a symmetric
# random variable with $\mathbb{E}\left[f^{\ell}\right]=0 .$ Moreover, we
# have the following identity
# $$\mathbb{E}\left[\left(f_{i}^{\ell}\right)^{2}\right]=\sum_{k} \mathbb{E}\left[\left(W_{i k}^{\ell}\right)^{2}\right] \mathbb{E}\left[\sigma\left(f_{k}^{\ell-1}\right)^{2}\right]$$
# Now, if $\sigma=i d$, we can prove by induction from $\ell=1$ that
# $$\mathbb{V}\left[f_{i}^{L}\right]=\left(\Pi_{\ell=2}^{L} n_{\ell-1} \operatorname{Var}\left[W_{s t}^{\ell}\right]\right)\left(\mathbb{V}\left[W_{s t}^{1}\right] \sum_{k} \mathbb{E}\left[\left([x]_{k}\right)^{2}\right]\right)$$
# We make this assumption that $\sigma=i d$, which is pretty reasonably
# since most activation functions in use at the time (such as the
# hyperbolic tangent) were close to the identity near 0 .
# 
# Now, if we set
# 
# $$
#     \mathbb{V}\left[W_{i k}^{\ell}\right]=\frac{1}{n_{\ell-1}}, \quad \forall \ell \geq 2
# $$
# 
# we will obtain
# 
# $$
#     \mathbb{V}\left[f_{i}^{L}\right]=\mathbb{V}\left[f_{j}^{L-1}\right]=\cdots=\mathbb{V}\left[f_{k}^{1}\right]=\mathbb{V}\left[W_{s t}^{1}\right] \sum_{k} \mathbb{E}\left[\left([x]_{k}\right)^{2}\right]
# $$
# 
# Thus, in pure DNN models, it is enough to just control
# $\sum_{k} \mathbb{E}\left[\left([x]_{k}\right)^{2}\right] .$
# 
# A similar analysis of the propagation of the gradient
# $\left(\frac{\partial L(\theta)}{\partial f^{t}}\right)$ suggests that
# we set 
# 
# $$
#     \mathbb{V}\left[W_{i k}^{\ell}\right]=\frac{1}{n_{\ell}}
# $$
# 
# Thus, the Xavier’s initialization suggests to initialize
# $W_{i k}^{\ell}$ with variance as:
# 
# -   To control $\mathbb{V}\left[f_{i}^{\ell}\right]:$
# 
# $$
#     \operatorname{Var}\left[W_{i k}^{\ell}\right]=\frac{1}{n_{\ell-1}}
# $$
# 
# -   To control
#     $\mathbb{V}\left[\frac{\partial L(\theta)}{\partial f_{i}^{l}}\right]:$
# 
# $$
#     \operatorname{Var}\left[W_{i k}^{\ell}\right]=\frac{1}{n_{\ell}}
# $$
# 
# -   Trade-off to control
#     $\mathbb{V}\left[\frac{\partial L(\theta)}{\partial W_{i k}^{l}}\right]:$
# 
# $$
#     \operatorname{Var}\left[W_{i k}^{\ell}\right]=\frac{2}{n_{\ell-1}+n_{\ell}}
# $$
# 
# Here we note that, this analysis works for all symmetric type
# distribution around zero, but we often just choose uniform distribution
# $\mathcal{U}(-a, a)$ and normal distribution
# $\mathcal{N}\left(0, s^{2}\right) .$ Thus, the final version of Xavier’s
# initialization takes the trade-off type as
# 
# $$
#     W_{i k}^{\ell} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\ell}+n_{\ell-1}}}, \sqrt{\frac{6}{n_{\ell}+n_{\ell-1}}}\right)
# $$
# 
# or
# 
# $$
#     W_{i k}^{\ell} \sim \mathcal{N}\left(0, \frac{2}{n_{\ell}+n_{\ell-1}}\right) 
# $$
# 
# ### Kaiming’s initialization
# 
# In \[2\], Kaiming He and others extended this analysis to get an exact
# result when the activation function is the ReLU.
# 
# We first have the following lemma for symmetric distribution.
# 
# Lemma 2. If $X_{i} \in \mathbb{R}$ for $i=1:$ n are i.i.d with symmetric
# probability density function $p(x)$, i.e. $p(x)$ is even. Then for any
# nonzero random vector
# $Y=\left(Y_{1}, Y_{2}, \cdots, Y_{n}\right) \in \mathbb{R}^{n}$ which is
# independent with $X_{i}$, the following random variable
# 
# $$
#     Z=\sum_{i=1}^{n} X_{i} Y_{i}
# $$
# 
# is also symmetric.
# 
# Then state the following result for ReLU function and random variable
# with symmetric distribution around 0 .
# 
# Lemma 3. If $X$ is a random variable on $\mathbb{R}$ with symmetric
# probability density $p(x)$ around zero, i.e., $$p(x)=p(-x)$$ Then we
# have $\mathbb{E} X=0$ and
# 
# $$
#     \mathbb{E}\left[[\operatorname{ReLU}(X)]^{2}\right]=\frac{1}{2} \operatorname{Var}[X]
# $$
# 
# Based on the previous Lemma 1, we know that $f_{k}^{\ell-1}$ is a
# symmetric distribution around 0 . The most important observation in
# Kaiming’s paper is that:
# 
# $$
#     \mathbb{V}\left[f_{i}^{\ell}\right]=n_{\ell-1} \mathbb{V}\left[W_{i j}^{\ell}\right] \mathbb{E}\left[\left[\sigma\left(f_{j}^{\ell-1}\right)\right]^{2}\right]=n_{\ell-1} \mathbb{V}\left[W_{i k}^{\ell}\right] \frac{1}{2} \mathbb{V}\left[f_{k}^{\ell-1}\right]
# $$
# 
# if $\sigma=$ ReLU. Thus, Kaiming’s initialization suggests to take:
# 
# $$
#     \mathbb{V}\left[W_{i k}^{\ell}\right]=\frac{2}{n_{\ell-1}}, \quad \forall \ell \geq 2
# $$
# 
# For the first layer $\ell=1$, by definition 
# 
# $$
#     f^{1}=W^{1} x+b^{1}
# $$
# 
# there is no ReLU, thus it should be
# $\mathbb{V}\left[W_{i k}^{1}\right]=\frac{1}{d} .$ For simplicity, they
# still use $\mathbb{V}\left[W_{i k}^{1}\right]=$ $\frac{2}{d}$ in the
# paper. Similarly, an analysis of the propagation of the gradient
# suggests that we set
# $\mathbb{V}\left[W_{i k}^{\ell}\right]=\frac{2}{n_{\ell}}$. However, in
# paper authors did not suggest to take the trade-off version, they
# just chose
# 
# $$
#     \mathbb{V}\left[W_{i k}^{\ell}\right]=\frac{2}{n_{\ell-1}}
# $$
# 
# as default.
# 
# Thus, the final version of Kaiming’s initialization takes the forward
# type as
# 
# $$
#     W_{i k}^{\ell} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\ell-1}}}, \sqrt{\frac{6}{n_{\ell-1}}}\right)
# $$
# 
# or
# 
# $$
#     W_{i k}^{\ell} \sim \mathcal{N}\left(0, \frac{2}{n_{\ell-1}}\right)
# $$
# 
# ### Initialization in CNN models and experiments
# 
# For CNN models, following the analysis above we have the next iterative
# scheme in CNNs
# 
# $$
#     f^{\ell, i}=K^{\ell, i} * \sigma\left(f^{\ell, i-1}\right)
# $$
# 
# where
# $f^{\ell, i-1} \in \mathbb{R}^{c_{\ell} \times n_{\ell} \times m_{\ell}}, f^{\ell, i} \in \mathbb{R}^{h_{\ell} \times n_{\ell} \times m_{\ell}}$
# and
# $K \in \mathbb{R}^{(2 k+1) \times(2 k+1) \times h_{\ell} \times c_{\ell}}$.
# Thus we have
# 
# $$
#     \left[f^{\ell, i}\right]_{h ; p, q}=\sum_{c=1}^{c_{l}} \sum_{s, t=-k}^{k} K_{h, c ; s, t}^{\ell, i} * \sigma\left(\left[f^{\ell, i-1}\right]_{c ; p+s, q+t}\right)
# $$
# 
# Take variance on both sides, we will get
# 
# $$
#     \mathbb{V}\left[\left[f^{\ell, i}\right]_{h ; p, q}\right]=c_{\ell}(2 k+1)^{2} \mathbb{V}\left[K_{h, o ; s, t}^{\ell, i}\right] \mathbb{E}\left[\left(\left[f^{\ell, i-1}\right]_{o ; p+s, q+t}\right)^{2}\right]
# $$
# 
# thus we have the following initialization strategies: Xavier’s
# initialization
# 
# $$
#     \mathbb{V}\left[K_{h, o ; s, t}^{\ell, i}\right]=\frac{2}{\left(c_{\ell}+h_{\ell}\right)(2 k+1)^{2}}
# $$
# 
# Kaiming’s initialization
# 
# $$
#     \mathbb{V}\left[K_{h, o ; s, t}^{\ell, i}\right]=\frac{2}{c_{\ell}(2 k+1)^{2}}
# $$
# 
# Here we can take this Kaiming’s initialization as:
# 
# -   Double the Xavier’s choice, and get
# 
# $$
#     \mathbb{V}\left[K_{h, o ; s, t}^{\ell, i}\right]=\frac{4}{\left(c_{\ell}+h_{\ell}\right)(2 k+1)^{2}} 
# $$
# 
# -   Then pick $c_{\ell}$ or $h_{\ell}$ for final result
# 
# $$
#     \mathbb{V}\left[K_{h, o ; s, t}^{\ell, i}\right]=\frac{4}{\left(c_{\ell}+h_{\ell}\right)(2 k+1)^{2}}=\frac{2}{c_{\ell}(2 k+1)^{2}} 
# $$
# 
# And they have the both uniform and normal distribution type.
# 
# ![image](images/img1.png)
# 
# Fig. The convergence of a 22-layer large model. The $x$-axis is the
# number of training epochs. The y-axis is the top-1 error of 3,000 random
# val samples, evaluated on the center crop. Use ReLU as the activation
# for both cases. Both Kaiming’s initialization (red) and “Xavier’s”
# (blue) lead to convergence, but Kaiming’s initialization starts
# reducing error earlier.
# 
# ![image](images/img2.png)
# 
# Fig. The convergence of a 30-layer small model (see the main text).
# Use ReLU as the activation for both cases. Kaiming’s initialization
# (red) is able to make it converge. But “Xavier’s” (blue) \[1\]
# completely stalls - It is also verified that that its gradients are all
# diminishing. It does not converge even given more epochs. Given a
# 22-layer model, in cifar10 the convergence with Kaiming’s initialization
# is faster than Xavier’s, but both of them are able to converge and the
# validation accuracies with two different initialization are about the
# same(error is $33.82,33.90)$.
# 
# With extremely deep model with up to 30 layers, Kaiming’s initialization
# is able to make the model convergence. On the contrary, Xavier’s method
# completely stalls the learning.

# In[ ]:




