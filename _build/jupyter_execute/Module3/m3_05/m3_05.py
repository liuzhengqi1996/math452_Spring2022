#!/usr/bin/env python
# coding: utf-8

# # Universal approximation properties

# In[1]:


from IPython.display import IFrame

IFrame(src= "https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_yk2026jt&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_to15zpm3",width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2022/01/C05_-Universal-approximation-properties.pdf)

# ## Approximation Properties of Neural Network Function Class
# 
# 
# ### Qualitative convergence results
# 
# ```{prf:theorem} (Universal Approximation Property of Shallow Neural Networks)
# :label: thm1
# Let $\sigma$ be a Riemann integrable function and
# $\sigma \in L_{l o c}^{\infty}(\mathbb{R}) .$ Then $\Sigma_{d}(\sigma)$
# in dense in $C(\Omega)$ for any compact $\Omega \subset \mathbb{R}^{n}$
# if and and only if $\sigma$ is not a polynomial!
# Namely, if $\sigma$ is not a polynomial, then, for any
# $f \in C(\bar{\Omega})$, there exists a sequence
# $\phi_{n} \in \mathrm{DNN}_{1}$ such that
# 
# $$
#     \max _{x \in \bar{\Omega}}\left|\phi_{n}(x)-f(x)\right| \rightarrow 0, \quad n \rightarrow \infty
# $$
# ```
# 
# ```{prf:proof} Let us first prove the theorem in a special case that
# $\sigma \in C^{\infty}(\mathbb{R}) .$ Since
# $\sigma \in C^{\infty}(\mathbb{R})$, it follows that for every
# $\omega, b$,
# 
# $$
#     \frac{\partial}{\partial \omega_{j}} \sigma(\omega \cdot x+b)=\lim _{n \rightarrow \infty} \frac{\sigma\left(\left(\omega+h e_{j}\right) \cdot x+b\right)-\sigma(\omega \cdot x+b)}{h} \in \bar{\Sigma}_{d}(\sigma)
# $$
# 
# for all $j=1, \ldots, d$.
# 
# By the same argument, for
# $\alpha=\left(\alpha_{1}, \ldots, \alpha_{d}\right)$
# 
# $$
#     D_{\omega}^{\alpha} \sigma(\omega \cdot x+b) \in \bar{\Sigma}_{d}(\sigma)
# $$
# 
# for all $k \in \mathbb{N}, j=1, \ldots, d, \omega \in \mathbb{R}^{d}$
# and $b \in \mathbb{R}$.
# 
# Now
# 
# $$
#     D_{\omega}^{\alpha} \sigma(\omega \cdot x+b)=x^{\alpha} \sigma^{(k)}(\omega \cdot x+b)
# $$
# 
# where $k=|\alpha|$ and
# $x^{\alpha}=x_{1}^{\alpha_{1}} \cdots x_{d}^{\alpha_{d}}$. Since
# $\sigma$ is not a polynomial there exists a $\theta_{k} \in \mathbb{R}$
# such that $\sigma^{(k)}\left(\theta_{k}\right) \neq 0$. Taking
# $\omega=0$ and $b=\theta_{k}$, we thus see that
# $x_{j}^{k} \in \bar{\Sigma}_{d}(\sigma) .$ Thus, all polynomials of the
# form $x_{1}^{k_{1}} \cdots x_{d}^{k_{d}}$ are in
# $\bar{\Sigma}_{d}(\sigma)$. This implies that $\bar{\Sigma}_{d}(\sigma)$
# contains all polynomials. By Weierstrass’s Theorem， it follows that $\bar{\Sigma}_{d}(\sigma)$
# contains $C(K)$ for each compact $K \subset \mathbb{R}^{n} .$ That is
# $\Sigma_{d}(\sigma)$ is dense in $C\left(\mathbb{R}^{d}\right) .$
# 
# Now we consider the case that $\sigma$ is only Riemann integrable.
# Consider the mollifier $\eta$
# 
# $$
#     \eta(x)=\frac{1}{\sqrt{\pi}} e^{-x^{2}}
# $$
# 
# Set $\eta_{\epsilon}=\frac{1}{\epsilon} \eta\left(\frac{x}{\epsilon}\right) .$
# Then consider $\sigma_{\eta_{\epsilon}}$
# 
# $$
#     \sigma_{\eta_{\epsilon}}(x):=\sigma * \eta_{\epsilon}(x)=\int_{\mathbb{R}} \sigma(x-y) \eta_{\epsilon}(y) d y
# $$ (eq1_4) 
# 
# for a given activation function $\sigma$
# It can be seen that
# $\sigma_{\eta_{\epsilon}} \in C^{\infty}(\mathbb{R}) .$ We first notice
# that
# $\bar{\Sigma}_{1}\left(\sigma_{\eta_{\epsilon}}\right) \subset \bar{\Sigma}_{1}(\sigma)$,
# which can be done easily by checking the Riemann sum of
# $\sigma_{\eta_{\epsilon}}(x)=\int_{\mathbb{R}} \sigma(x-y) \eta_{\epsilon}(y) d y$
# is in $\bar{\Sigma}_{1}(\sigma)$.
# 
# Following the argument in the beginning of the proof proposition, we
# want to show that
# $\left.\bar{\Sigma}_{1}\left(\sigma_{\eta_{\epsilon}}\right)\right)$
# contains all polynomials. For this purpose, it suffices to show that
# there exists $\theta_{k}$ and $\sigma_{\eta_{\epsilon}}$ such that
# $\sigma_{\eta_{\epsilon}}^{(k)}\left(\theta_{k}\right) \neq 0$ for each
# $\mathrm{k}$. If not, then there must be $k_{0}$ such that
# $\sigma_{\eta_{\epsilon}}^{\left(k_{0}\right)}(\theta)=0$ for all
# $\theta \in \mathbb{R}$ and all $\epsilon>0$. Thus
# $\sigma_{\eta_{\epsilon}}$ ’s are all polynomials with degree at most
# $k_{0}-1 .$ In particular, It is known that
# $\eta_{\epsilon} \in C_{0}^{\infty}(\mathbb{R})$ and
# $\sigma * \eta_{\epsilon}$ uniformly converges to $\sigma$ on compact
# sets in $\mathbb{R}$ and $\sigma * \eta_{\epsilon}$ ’s are all
# polynomials of degree at most $k_{0}-1 .$ Polynomials of a fixed degree
# form a closed linear subspace, therefore $\sigma$ is also a polynomial
# of degree at most $k_{0}-1$, which leads to contradiction.
# ```
# 
# ### Properties of polynomials using Fourier transform
# 
# 
# We make use of the theory of tempered distributions and
# we begin by collecting some results of independent interest, which will
# also be important later. We begin by noting that an activation function
# $\sigma$ which satisfies a polynomial growth condition
# $|\sigma(x)| \leq C(1+|x|)^{n}$ for some constants $C$ and $n$ is a
# tempered distribution. As a result, we make this assumption on our
# activation functions in the following theorems. We briefly note that
# this condition is sufficient, but not necessary (for instance an
# integrable function need not satisfy a pointwise polynomial growth
# bound) for $\sigma$ to be represent a tempered distribution.
# 
# We begin by studying the convolution of $\sigma$ with a Gaussian
# mollifier. Let $\eta$ be a Gaussian mollifier
# 
# $$
#     \eta(x)=\frac{1}{\sqrt{\pi}} e^{-x^{2}}
# $$
# 
# Set $\eta_{\epsilon}=\frac{1}{\epsilon} \eta\left(\frac{x}{\epsilon}\right) .$
# Then consider $\sigma_{\epsilon}$
# 
# $$
#     \sigma_{\epsilon}(x):=\sigma * \eta_{\epsilon}(x)=\int_{\mathbb{R}} \sigma(x-y) \eta_{\epsilon}(y) d y
# $$
# 
# for a given activation function $\sigma$.
# 
# It is clear that $\sigma_{\epsilon} \in C^{\infty}(\mathbb{R}) .$
# Moreover, by considering the Fourier transform (as a tempered
# distribution) we see that
# 
# $$
#     \hat{\sigma}_{\epsilon}=\hat{\sigma} \hat{\eta}_{\epsilon}=\hat{\sigma} \eta_{\epsilon^{-1}}
# $$
# 
# We begin by stating a lemma which characterizes the set of polynomials
# in terms of their Fourier transform.
# 
# ```{prf:lemma}
# Given a tempered distribution $\sigma$, the following
# statements are equivalent:
# 
# 1.  $\sigma$ is a polynomial
# 
# 2.  $\sigma_{\epsilon}$ given by {eq}`eq1_4` is a polynomial for any
#     $\epsilon>0$.
# 
# 3.  $\operatorname{supp}(\hat{\sigma}) \subset\{0\}$.
# ```
# ```{prf:proof}
# We begin by proving that (3) and (1) are equivalent. This follows
# from a characterization of distributions supported at a single point. In particular, a
# distribution supported at 0 must be a finite linear combination of Dirac
# masses and their derivatives. In particular, if $\hat{\sigma}$ is
# supported at 0 , then
# 
# $$
#     \hat{\sigma}=\sum_{i=1}^{n} a_{i} \delta^{(i)} 
# $$
# 
# Taking the inverse Fourier transform and noting that the inverse Fourier transform of
# $\delta^{(i)}$ is $c_{i} x^{i}$, we see that $\sigma$ is a polynomial.
# This shows that (3) implies (1), for the converse we simply take the
# Fourier transform of a polynomial and note that it is a finite linear
# combination of Dirac masses and their derivatives.
# 
# Finally, we prove the equivalence of (2) and (3). For this it suffices
# to show that $\hat{\sigma}$ is supported at 0 iff
# $\hat{\sigma}_{\epsilon}$ is supported at $0 .$ This follows from
# equation $1.5$ and the fact that $\eta_{\epsilon^{-1}}$ is nowhere
# vanishing.
# ```
# 
# 
# 

# In[ ]:




