#!/usr/bin/env python
# coding: utf-8

# # Batch normalization

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_c20ifjjk&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_t1lki6t4"  ,width='800', height='500')


# ### Recall the original DNN model
# 
# Consider the classical (fully connected) artificial deep neural network
# (DNN) $f^{L}$,
# $$\begin{cases}f^{1} & =\theta^{1}(x):=W^{1} x+b^{1}, \\ f^{\ell} & =\theta^{\ell} \circ \sigma\left(f^{\ell-1}\right):=W^{\ell} \sigma\left(f^{\ell-1}\right)+b^{\ell}, \ell=2, \ldots, L .\end{cases}$$
# where $x \in \mathbb{R}^{n}$ is the input vector, $\sigma$ is a
# non-linear function (activation).
# 
# 2 “’Real’” Batch Normalization and "’new’ model
# -----------------------------------------------
# 
# Definition of $B N$ operation based on the batch
# ================================================
# 
# Following the idea in normalization, we consider that we have the all
# training data as $$(X, Y):=\left\{x_{i}, y_{i}\right\}_{i=1}^{N} .$$
# Since the normalization is applied to each activation independently, let
# us focus on a particular activation $\left[f^{\ell}\right]_{k}$ and omit
# $k$ as $f^{\ell}$ for clarity. We have $N$ values of this activation in
# the batch, $$X=\left\{x_{1}, \cdots, x_{N}\right\}$$ Let the normalized
# values be $\hat{f}^{\ell}$, and their linear transformations be
# $\tilde{f}^{\ell} .$ $$\begin{gathered}
# \mu_{X}^{\ell} \leftarrow \mathbb{E}_{x \sim X}\left[f^{\ell}(x)\right]=\frac{1}{N} \sum_{i=1}^{N} f^{\ell}\left(x_{i}\right) \\
# \sigma_{X}^{\ell} \leftarrow \mathbb{E}_{x \sim X}\left[\left(f^{\ell}(x)-\mathbb{E}_{x \sim X}\left[f^{\ell}(x)\right]\right)^{2}\right]=\frac{1}{N} \sum_{i=1}^{N}\left(f^{\ell}\left(x_{i}\right)-\mu_{X}\right)^{2} \quad \text { batch mean } \\
# \hat{f}^{\ell}(x) \leftarrow \frac{f^{\ell}(x)-\mu_{X}^{\ell}}{\sqrt{\sigma_{X}^{\ell}+\epsilon}} \\
# \tilde{f}^{\ell}(x) \leftarrow \gamma^{\ell} \hat{f}^{\ell}(x)+\beta^{\ell}
# \end{gathered}$$ Here we note that all these operations in the previous
# equation are defined by element-wise. Then at last, we define the BN
# operation based on the batch set as
# $$\mathrm{BN}_{X}\left(f^{\ell}(x)\right)=\tilde{f}^{\ell}(x):=\gamma^{\ell} \frac{f^{\ell}(x)-\mu_{X}^{\ell}}{\sqrt{\sigma_{X}^{\ell}+\epsilon}}+\beta^{\ell}$$
# where $\tilde{f}^{\ell}(x), \mu_{X}^{\ell}$ and $\sigma_{X}^{\ell}$ are
# given above.
# 
# “New” model for BN
# ==================
# 
# In summary, we have the new DNN model with BN as:
# $$\begin{cases}\tilde{f}^{1}\left(x_{i}\right) & =\left(\theta^{1}\left(x_{i}\right)\right) \\ \tilde{f}^{\ell} & =\theta^{\ell} \circ \sigma \circ \mathrm{BN}_{X}\left(\tilde{f}^{\ell-1}\right), \quad \ell=2, \ldots, L .\end{cases}$$
# For a more comprehensive notation, we can use the next notation
# $$\sigma_{\mathrm{BN}}:=\sigma \circ \mathrm{BN}_{X}$$ Here one thing is
# important that we need to mention is that because of the new scale
# $\gamma^{\ell}$ and shift $\beta^{\ell}$ added after the BN operation.
# We can remove the basis $b^{\ell}$ in $\theta^{\ell}$, thus to say the
# real model we will compute should be
# $$\begin{cases}\tilde{f}^{1}\left(x_{i}\right) & =W^{1} x_{i} \\ \tilde{f}^{\ell} & =W^{\ell} \sigma_{\mathrm{BN}}\left(\tilde{f}^{\ell-1}\right), \quad \ell=2, \ldots, L .\end{cases}$$
# Combine the two definition, we note
# $$\tilde{\Theta}:=\{W, \gamma, \beta\}$$ where
# $W=\left\{W^{1}, \cdots, W^{l}\right\}, \gamma:=\left\{\gamma^{2}, \cdots, \gamma^{L}\right\}$
# and $\beta:=\left\{\beta^{2}, \cdots, \beta^{L}\right\}$
# 
# Finally, we have the loss function as:
# $$\mathcal{L}(\tilde{\Theta})=\mathbb{E}_{(x, y) \sim(X, Y)} \approx \frac{1}{N} \sum_{i=1}^{N} \ell\left(\tilde{f}^{L}\left(x_{i} ; \tilde{\Theta}\right), y_{i}\right)$$
# A key observation in $(1.57)$ and the new BN model $(1.55)$ is that
# $$\begin{aligned}
# \mu_{X}^{\ell} &=\mathbb{E}_{x \sim X}\left[f^{\ell}(x)\right] \\
# \sigma_{X}^{\ell} &=\mathbb{E}_{x \sim X}\left[\left(f^{\ell}(x)-\mathbb{E}_{x \sim X}\left[f^{\ell}(x)\right]\right)^{2}\right] \\
# \mathcal{L}(\tilde{\Theta}) &=\mathbb{E}_{(x, y) \sim(X, Y)}\left[\ell\left(\tilde{f}^{L}\left(x_{i} ; \tilde{\Theta}\right), y_{i}\right)\right]
# \end{aligned}$$ Here we need to mention that $$x \sim X$$ means $x$
# subject to the discrete distribution of all data $X$.
# 
# ### BN: some ’modified” SGD on new batch normalized model
# 
# Following the key observation in (1.58), and recall the similar case in
# SGD, we do the the sampling trick in (1.57) and obtain the mini-batch
# SGD: $$x \sim X \approx x \sim \mathcal{B},$$ here $\mathcal{B}$ is a
# mini-batch of batch $X$ with $\mathcal{B} \subset X .$
# 
# However, for problem in (1.57), it is very difficult to find some subtle
# sampling method because of the composition of $\mu_{X}^{\ell}$ and
# $\left[\sigma_{X}^{\ell}\right]^{2}$. However, one simple way for
# sampling (1.57) can be chosen as taking (1.59) for all the expectation
# case in (1.57) and (1.58).
# 
# This is to say, in training process ( $t$-th step for example), once we
# choose $B_{t} \subset X$ as the mini-batch, then the model becomes
# $$\begin{cases}\tilde{f}^{1}\left(x_{i}\right) & =W^{1} x_{i}, \\ \tilde{f}^{\ell} & =W^{\ell} \sigma_{\mathrm{BN}}\left(\tilde{f}^{\ell-1}\right), \quad \ell=2, \ldots, L .\end{cases}$$
# where
# $$\sigma_{\mathrm{BN}}:=\sigma \circ \mathrm{BN}_{\mathcal{B}_{t}},$$ or
# we can say that $X$ is replaced by $\mathcal{B}_{t}$ in this case.
# 
# Here $\mathrm{BN}_{\mathcal{B}_{t}}$ is defined by $$\begin{array}{cr}
# \mu_{\mathcal{B}_{t}}^{\ell} & \leftarrow \frac{1}{m} \sum_{i=1}^{m} f^{\ell}\left(x_{i}\right) \\
# \sigma_{\mathcal{B}_{t}}^{\ell} & \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(f^{\ell}\left(x_{i}\right)-\mu_{\mathcal{B}_{t}}\right)^{2} \quad \text { mini-batch mean } \\
# \hat{f}^{\ell}(x) & \leftarrow \frac{f^{\ell}(x)-\mu_{\mathcal{B}_{t}}^{\ell}}{\sqrt{\sigma_{\mathcal{B}_{t}}^{\ell}+\epsilon}} \\
# \mathrm{BN}_{\mathcal{B}_{t}}\left(\tilde{f}^{\ell}\right):=\tilde{f}^{\ell}(x) & \leftarrow \gamma^{\ell} \hat{f}^{\ell}(x)+\beta^{\ell} \\
# & \text { normalize }
# \end{array}$$ Here BN operation introduce some new parameters as
# $\gamma$ and $\beta$. Thus to say, for training phase, if we choose
# mini-batch as $\mathcal{B}_{t}$ in $t$-th training step, we need to take
# gradient as
# $$\frac{1}{m} \nabla_{\tilde{\Theta}} \sum_{i \in \mathcal{B}_{t}} \ell\left(\tilde{f}^{L}\left(x_{i} ; \tilde{\Theta}\right), y_{i}\right)$$
# which needs us the to take gradient for $\mu_{B}^{\ell}$ or
# $\left[\sigma_{B}^{\ell}\right]^{2}$ w.r.t $w^{i}$ for $i \leq \ell$.
# 
# Questions: To derive the new gradient formula for BN step because of the
# fact that
# $$\mu_{\mathcal{B}_{t}}^{\ell}, \quad \text { and } \quad \sigma_{\mathcal{B}_{t}}^{\ell}$$
# contain the output of $\tilde{f}^{\ell-1}$.
# 
# This is exact the batch normalization method described in \[3\].
# 
# ### Testing phase in Batch-Normalized DNN
# 
# One key problem is that, in the BN operator, we need to compute the mean
# and variance in a data set (batch or mini-batch). However, in the
# inference step, we just input one data into this DNN, how to compute the
# BN operator in this situation.
# 
# Actually, the $\gamma$ and $\beta$ parameter is fixed after training,
# the only problem is to compute the mean $\mu$ and variance $\sigma^{2}$.
# All the mean $\mu_{\mathcal{B}_{t}}$ and variance
# $\sigma_{\mathcal{B}}^{2}$ during the training phase are just the
# approximation of the mean and variance of whole batch i.e. $\mu_{X}$ and
# $\sigma_{X}^{2}$ as shown in (1.58).
# 
# One natural idea might be just use the BN operator w.r.t to the whole
# training data set, thus to say just compute $\mu_{X}$ and
# $\sigma_{X}^{2}$ by definition in (1.51).
# 
# However, there are at least the next few problems:
# 
# -   computation cost,
# 
# -   ignoring the statistical approximation (don’t make use of the
#     $\mu_{\mathcal{B}_{t}}$ and $\sigma_{\mathcal{B}_{t}}^{2}$ in
#     training phase).
# 
# Considering that we have the statistical approximation for $\mu_{X}$ and
# $\sigma_{X}^{2}$ during each SGD step, moving average might be a more
# straightforward way. Thus two say, we define the $\mu^{\ell}$ and
# $\left[\sigma^{\ell}\right]^{2}$ for the inference (test) phase as
# $$\mu^{\ell}=\frac{1}{T} \sum_{t=1}^{T} \mu_{\mathcal{B}_{t}}^{\ell}, \quad \sigma^{\ell}=\frac{1}{T} \frac{m}{m-1} \sum_{t=1}^{T} \sigma_{\mathcal{B}_{t}}^{\ell}$$
# Here we take Bessel’s correction for unbiased variance. The above moving
# average step is found in the original paper of BN in \[3\].
# 
# Another way to do this is to call the similar idea in momentum. At each
# time step we update the running averages for mean and variance using an
# exponential decay based on the momentum parameter: $$\begin{aligned}
# &\mu_{\mathcal{B}_{t}}^{\ell}=\alpha \mu_{\mathcal{B}_{t-1}}^{\ell}+(1-\alpha) \mu_{\mathcal{B}_{t}}^{\ell} \\
# &\sigma_{\mathcal{B}_{t}}^{\ell}=\alpha \sigma_{\mathcal{B}_{t-1}}^{\ell}+(1-\alpha) \sigma_{\mathcal{B}_{t}}^{\ell}
# \end{aligned}$$ $\alpha$ is close to 1 , we can take it as $0.9$
# generally. Then we all take bath mean and variance as
# $\mu_{X}^{\ell} \approx \mu_{\mathcal{B}_{T}}^{\ell}$ and
# $\sigma_{X}^{\ell} \approx \sigma_{\mathcal{B}_{T}}^{\ell} .$
# 
# Many people argue that the variance here should also use Bessel’s
# correction.
# 
# ### Batch Normalization for CNN
# 
# One key idea in $\mathrm{BN}$ is to do normalization with each scalar
# features (neurons) separately along a mini-batch. Thus to say, we need
# one to identify what is neuron in CNN. This is a historical problem,
# some people think neuron in CNN should be the pixel in each channel some
# thing that each channel is just one neuron. BN choose the later one. One
# (most ?) important reason for this choice is the fact of computation
# cost. For convolutional layers, BN additionally wants the normalization
# to obey the convolutional property - so that different elements of the
# same feature map, at different locations, are normalized in the same
# way. To compute $\mu_{\mathcal{B}_{t}}^{\ell}$, we take mean of the set
# of all values in a feature map across both the elements of a mini-batch
# and spatial locations - so for a mini-batch of size $m$ and feature maps
# of size $m_{\ell} \times n_{\ell}$ (image geometrical size), we use the
# effective mini-batch of size $m m_{\ell} n_{\ell}$. We learn a pair of
# parameters $\gamma_{k}$ and $\beta_{k}$ per feature map (k-th channel),
# rather than per activation
# 
# For simplicity, then have the following BN scheme for CNN
# $$\begin{array}{cc}
# {\left[\mu_{\mathcal{B}_{t}}^{\ell}\right]_{j} \leftarrow \frac{1}{m \times m_{\ell} \times n_{\ell}} \sum_{i=1}^{m} \sum_{1 \leq s \leq m_{\ell}, 1 \leq t \leq n_{\ell}}\left[f^{\ell}\left(x_{i}\right)\right]_{j ; s t}} & \text { mean on channel } j \\
# {\left[\sigma_{\mathcal{B}_{t}}^{\ell}\right]_{j} \leftarrow \frac{1}{m \times m_{\ell} \times n_{\ell}} \sum_{i=1}^{m} \sum_{1 \leq s \leq m_{\ell}, 1 \leq t \leq n_{\ell}}\left(\left[f^{\ell}\left(x_{i}\right)\right]_{j ; s t}-\left[\mu_{\mathcal{B}_{t}}^{\ell}\right]_{j}\right)^{2}} & \text { variance on channel } j \\
# {\left[\hat{f}^{\ell}(x)\right]_{j ; s t} \leftarrow \frac{\left[f^{\ell}(x)\right]_{j, s t}-\left[\mu_{\mathcal{B}_{t}}^{\ell}\right]_{j}}{\sqrt{\left[\sigma_{\mathcal{B}_{t}}^{\ell}\right]_{j}+\epsilon}}} & \text { normalize }
# \end{array}$$
# $$\left[\mathrm{BN}_{\mathcal{B}_{t}}\left(\tilde{f}^{\ell}\right)\right]_{j ; s t}:=\left[\tilde{f}^{\ell}(x)\right]_{j ; s t} \leftarrow\left[\gamma^{\ell}\right]_{j}\left[\hat{f}^{\ell}(x)\right]_{j ; s t}+\left[\beta^{\ell}\right]_{j}$$
# scale and shift on channel

# In[ ]:




