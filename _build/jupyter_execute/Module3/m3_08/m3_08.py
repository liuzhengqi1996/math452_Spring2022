#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo Methods

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_r8xtoqvb&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_yh08ji56" ,width='800', height='500')


# In[2]:


#IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_o6yt8ku9&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_a5lk2qct" ,width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2022/01/C08_-Monte-Carlo-Methods.pdf)

# ## Monte Carlo methods
# Let $\lambda \geq 0$ be a probability density function on
# $G \subset \mathbb{R}^{d}$ such that
# 
# $$
#     \int_{G} \lambda(\omega) d \omega=1 
# $$
# 
# For example:
# 
# $$
#     \lambda(\omega)=\frac{1}{|G|}
# $$
# 
# if $G$ is bounded. The expectation is defined: 
# 
# $$
#     \mathbb{E} g:=\int_{G} g(\omega) \lambda(\omega) d \omega
# $$
# 
# and for any
# $h=h\left(\omega_{1}, \omega_{2}, \ldots, \omega_{n}\right): G \times G \cdots G \mapsto \mathbb{R}$
# 
# $$
#     \mathbb{E}_{n} h:=\int_{G \times G \times \ldots \times G} h\left(\omega_{1}, \omega_{2}, \ldots, \omega_{n}\right) \lambda\left(\omega_{1}\right) \lambda\left(\omega_{2}\right) \ldots \lambda\left(\omega_{n}\right) d \omega_{1} d \omega_{2} \ldots d \omega_{n}
# $$
# 
# ### A basic result
# ```{prf:lemma}
# :label: lemma1
# 
# For any $g \in L^{\infty}(G)$, we have
# 
# $$
#     \mathbb{E}_{n}\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2}=\left\{\begin{array}{c}
# \frac{1}{n} \mathbb{E}\left((\mathbb{E} g-g)^{2}\right) \leq \frac{1}{n} \sup _{\omega, \omega \in G}\left|g(\omega)-g\left(\omega^{\prime}\right)\right|^{2} \\
# \frac{1}{n}\left(\mathbb{E}\left(g^{2}\right)-(\mathbb{E}(g))^{2}\right) \leq \frac{1}{n} \mathbb{E}\left(g^{2}\right) \leq \frac{1}{n}\|g\|_{L^{\infty}}^{2},
# \end{array}\right.
# $$
# 
# ```
# ```{prf:proof}
# First note that 
# 
# $$
#     \begin{aligned}
# \left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2} &=\frac{1}{n^{2}}\left(n \mathbb{E} g-\sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2}=\frac{1}{n^{2}}\left(\sum_{i=1}^{n}\left(\mathbb{E} g-g\left(\omega_{i}\right)\right)\right)^{2} \\
# &=\frac{1}{n^{2}} \sum_{i, j=1}^{n}\left(\mathbb{E} g-g\left(\omega_{i}\right)\right)\left(\mathbb{E} g-g\left(\omega_{j}\right)\right) \\
# &=\frac{I_{1}}{n^{2}}+\frac{I_{2}}{n^{2}}
# \end{aligned}
# $$
# 
# with
# 
# $\left.I_{1}=\sum_{i=1}^{n}\left(\mathbb{E} g-g\left(\omega_{i}\right)\right)^{2}, \quad I_{2}=\sum_{i \neq j}^{n}\left((\mathbb{E} g)^{2}-\mathbb{E}(g)\left(g\left(\omega_{i}\right)+g\left(\omega_{j}\right)\right)+g\left(\omega_{i}\right) g\left(\omega_{j}\right)\right)\right)$
# 
# Consider $I_{1}$, for any $i$,
# 
# $$
#     \mathbb{E}_{n}\left(\mathbb{E} g-g\left(\omega_{i}\right)\right)^{2}=\mathbb{E}_{n}(\mathbb{E} g-g)^{2}
# $$
# 
# Thus,
# 
# $$
#     \mathbb{E}_{n} I_{1}=n \mathbb{E}\left((\mathbb{E} g-g)^{2}\right)
# $$
# 
# For $I_{2}$, note that
# 
# $$
#     \mathbb{E}_{n} g\left(\omega_{i}\right)=\mathbb{E}_{n} g\left(\omega_{j}\right)=\mathbb{E}(g)$$
# and, for $i \neq j$, 
# 
# $$
#     \begin{aligned}
# \mathbb{E}_{n}\left(g\left(\omega_{i}\right) g\left(\omega_{j}\right)\right) &=\int_{G \times G \times \ldots \times G} g\left(\omega_{j}\right) g\left(\omega_{j}\right) \lambda\left(\omega_{1}\right) \lambda\left(\omega_{2}\right) \ldots \lambda\left(\omega_{n}\right) d \omega_{1} d \omega_{2} \ldots d \omega_{n} \\
# &=\int_{G \times G} g\left(\omega_{j}\right) g\left(\omega_{j}\right) \lambda\left(\omega_{1}\right) \lambda\left(\omega_{1}\right) \lambda\left(\omega_{2}\right) d \omega_{1} d \omega_{2} \\
# &=\mathbb{E}_{n}\left(g\left(\omega_{i}\right)\right) \mathbb{E}_{n}\left(g\left(\omega_{j}\right)\right)=[\mathbb{E}(g)]^{2}
# \end{aligned}
# $$
# 
# Thus
# 
# $$
#     \mathbb{E}_{n}\left(I_{2}\right)=\mathbb{E}_{n}\left(\sum_{i \neq j}^{n}\left((\mathbb{E} g)^{2}-\mathbb{E}(g)\left(\mathbb{E}\left(g\left(\omega_{i}\right)\right)+\mathbb{E}\left(g\left(\omega_{j}\right)\right)\right)+\mathbb{E}\left(g\left(\omega_{i}\right) g\left(\omega_{j}\right)\right)\right)\right)=0
# $$
# 
# Consequently, there exist the following two formulas for
# $\mathbb{E}_{n}\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2}:$
# 
# $$
#     \mathbb{E}_{n}\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2}=\frac{1}{n^{2}} \mathbb{E}_{n} I_{1}=\left\{\begin{array}{c}
# \frac{1}{n} \mathbb{E}\left((\mathbb{E} g-g)^{2}\right) \\
# \frac{1}{n}\left(\mathbb{E}\left(g^{2}\right)-(\mathbb{E} g)^{2}\right)
# \end{array}\right.
# $$
# 
# Based on the first formula above, since
# 
# $$
#     |g(\omega)-\mathbb{E} g|=\left|\int_{G}(g(\omega)-g(\tilde{\omega})) \lambda(\tilde{\omega}) d \tilde{\omega}\right| \leq \sup _{\omega, \omega \in G}\left|g(\omega)-g\left(\omega^{\prime}\right)\right|
# $$
# 
# it holds that
# 
# $$
#     \mathbb{E}_{n}\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2} \leq \frac{1}{n} \sup _{\omega, \omega \in G}\left|g(\omega)-g\left(\omega^{\prime}\right)\right|^{2}
# $$
# 
# Due to the second formula above,
# 
# $$
#     \mathbb{E}_{n}\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2} \leq \frac{1}{n} \mathbb{E}\left(g^{2}\right) \leq \frac{1}{n}\|g\|_{L^{\infty}}^{2}
# $$
# 
# which completes the proof. $\square$
# 
# Of course, we can use this to prove a high probability result.
# 
# ```{prf:corollary}
# Under the assumptions of the preceding lemma, we have
# 
# $$
#     \overline{\mathbb{P}}\left[\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2}>\frac{k}{n}\|g\|_{L^{\infty}}^{2}\right]<\frac{1}{k}
# $$
# 
# ```
# 
# ```{prf:proof}
# 
# $
#     \overline{\mathbb{P}}\left[\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2}>\epsilon\right] \leq \epsilon^{-1} \overline{\mathbb{E}}\left(\mathbb{E} g-\frac{1}{n} \sum_{i=1}^{n} g\left(\omega_{i}\right)\right)^{2} \leq \frac{1}{n \epsilon}\|g\|_{L^{\infty}}^{2}
# $
# 
# ```
# 
# This corollary implies that the set of $\omega_{i}$ where the estimate
# $n^{-1} \sum_{i=1}^{n} g\left(\omega_{i}\right)$ is far from the desired
# value $\mathbb{E} g$ is small.
# 
# The practical usefulness of this algorithm depends upon the existence of
# a repeatable process (for instance some physical process) which
# generates $\omega$ according to a desired distribution $\mu$.
# 
# The precise meaning of this last statement is essentially that the
# strong law of large numbers holds. Specifically, if
# $\omega_{1}, \ldots, \omega_{n}, \ldots$ is a infinite sequence
# generated by the process, and $A \subset \Omega$ is any a measurable
# set, then
# 
# $$
#     \lim _{n \rightarrow \infty} \frac{1}{n} \sum_{i=1}^{n} \chi_{A}\left(\omega_{i}\right)=\mu(A) 
# $$
# 
# Generating $n$ independent samples means generating
# $\omega_{1}, \ldots, \omega_{n}$ from $\mu^{n}$ according to the above
# notion. The existence of a realizable process generating samples from a
# probability distribution, and the practical use of such processes is an
# interesting topic in the intersection of statistics, physics, and
# computer science. In addition, statistics/probability theory studies how
# to take samples from one probability distribution and transform them to
# samples from another distribution.
# 
# ### Application
# ```{prf:lemma}
# :label: lemma2
# Let
# 
# $$
#     f(x)=\int_{G} g(x, \theta) \lambda(\theta) d \theta=\mathbb{E}(g)
# $$
# 
# with $\lambda(\theta) \geq 0$ and $\|\lambda(\theta)\|_{L^{1}(G)}=1$.
# For any $n \geq 1$, there exist $\theta_{i}^{*} \in G$ such that
# 
# $$
#     \left\|f-f_{n}\right\|_{L^{2}(\Omega)}^{2} \leq \frac{1}{n} \int_{G}\|g(\cdot, \theta)\|_{L^{2}(\Omega)}^{2} \lambda(\theta) d \theta=\frac{1}{n} \mathbb{E}\left(\|g(\cdot, \theta)\|_{L^{2}(\Omega)}^{2}\right)
# $$
# 
# where
# $\|g(\cdot, \theta)\|_{L^{2}(\Omega)}^{2}=\int_{\Omega}[g(x, \theta)]^{2} d \mu(x)$,
# and
# 
# $$
#     f_{n}(x)=\frac{1}{n} \sum_{i=1}^{n} g\left(x, \theta_{i}^{*}\right)
# $$
# 
# ```
# 
# ```{prf:proof}
# Introducing a probability distribution $\lambda(\theta)$,
# 
# $$
#     f(x)=\mathbb{Z}(g)
# $$
# 
# By {prf:ref}`lemma1`,
# 
# $$
#     \left.\mathbb{E}_{n}\left(\left(\mathbb{E}(g(x, \cdot))-\frac{1}{n} \sum_{i=1}^{n} g\left(x, \theta_{i}\right)\right)\right)^{2}\right) \leq \frac{1}{n} \mathbb{E}\left(g^{2}\right)
# $$
# 
# and
# 
# $$
#     \mathbb{E}_{n}\left(h\left(\theta_{1}, \theta_{2}, \cdots, \theta_{n}\right)\right) \leq \frac{1}{n} \mathbb{E}\left(\int_{\Omega} g^{2} d \mu(x)\right),
# $$
# 
# by taking integral where
# 
# $$
#     \left.h\left(\theta_{1}, \theta_{2}, \cdots, \theta_{n}\right)=\int_{\Omega}\left(\mathbb{E}(g(x, \cdot))-\frac{1}{n} \sum_{i=1}^{n} g\left(x, \theta_{i}\right)\right)\right)^{2} d \mu(x)
# $$
# 
# Sine $\mathbb{E}_{n}(1)=1$ and
# $\mathbb{E}_{n}(h) \leq \frac{1}{n} \mathbb{E}\left(\int_{O} g^{2} d \mu(x)\right)$,
# there exists
# $\left(\theta_{1}^{*}, \theta_{2}^{*}, \cdots, \theta_{n}^{*}\right) \in G \times G \times$
# $\cdots \times G$ such that
# 
# $$
#     h\left(\theta_{1}^{*}, \theta_{2}^{*}, \cdots, \theta_{n}^{*}\right) \leq \frac{1}{n} \int_{\Omega} \mathbb{E}\left(g^{2}\right) d \mu(x)
# $$
# 
# Otherwise,
# $\mathbb{E}_{n}(h)>\frac{1}{n} \mathbb{E}\left(\int_{\Omega} g^{2} d \mu(x)\right)$
# if
# $\left.h\left(\theta_{1}, \theta_{2}, \cdots, \theta_{n}\right)>\frac{1}{n} \int_{\Omega} \mathbb{E}\left(g^{2}\right)\right) d \mu(x) .$
# This implies that
# $$\left\|f-f_{n}\right\|_{L^{2}(\Omega)}^{2} \leq \frac{1}{n} \int_{G}\|g(\cdot, \theta)\|_{L^{2}(\Omega)}^{2} \lambda(\theta) d \theta,$$
# which completes the proof.
# ```
# 
# We also have a more general version of the above lemma.
# 
# ```{prf:lemma}
# :label: lemma3
# Let
# 
# $$
#     f(x)=\int_{G} g(x, \theta) \lambda(\theta) d \theta=\mathbb{E}(g)
# $$
# 
# with $\|\lambda(\theta)\|_{L^{1}(\Theta)}=1$. For any $n \geq 1$, there
# exist $\theta_{i}^{*} \in G$ such that
# 
# $$
#     \left\|f-f_{n}\right\|_{H^{m}(\Omega)}^{2} \leq \int_{G}\|g(\cdot, \theta)\|_{H^{m}(\Omega)}^{2} \lambda(\theta) d \theta=\frac{1}{n} \mathbb{E}\left(\|g(\cdot, \theta)\|_{H^{m}(\Omega)}^{2}\right)
# $$
# 
# where
# 
# $$
#     f_{n}(x)=\frac{1}{n} \sum_{i=1}^{n} g\left(x, \theta_{i}^{*}\right)
# $$
# 
# In particular, if
# 
# $$
#     \left|D^{\alpha} g(x, \theta)\right| \leq C, \quad \forall x, \theta,|\alpha| \leq m
# $$
# 
# Then 
# 
# $$
#     \left\|f-f_{n}\right\|_{H^{m}(\Omega)} \leq\left(\begin{array}{c}
# m+d \\
# m
# \end{array}\right)^{1 / 2}|\Omega|^{1 / 2} n^{-1 / 2}
# $$
# 
# For any
# $f(x)=\int_{G} g(x, \theta) \rho(\theta) d \theta$ with
# $\|\rho\|_{L^{1}(\Theta)} \neq 1 .$ Let
# $\lambda(\theta)=\frac{\rho(\theta)}{\|\rho\|_{L^{1}(\theta)}} .$ Thus,
# 
# $$
#     f(x)=\|\rho\|_{L^{1}(\Theta)} \int_{G} g(x, \theta) \lambda(\theta) d \theta
# $$
# 
# with $\|\lambda(\theta)\|_{L^{1}(\Theta)}=1 .$ We can apply the above
# two lemmas to the given function $f(x)$.
# 
# 
# ## Integral representations of functions
# 
# 
# ### Fourier representation
# 
# Consider the Fourier transform:
# 
# $$
#     \hat{f}(\omega)=\frac{1}{(2 \pi)^{d}} \int_{\mathbb{R}^{d}} e^{-i \omega \cdot x} f(x) d x \quad \forall \omega \in \mathbb{R}^{d}
# $$
# 
# We write $\hat{f}(\omega)=e^{i \theta(\omega)}|\hat{f}(\omega)| .$ By
# Fourier inversion formula,
# 
# $$
#     f(x)=\int_{\mathbb{R}^{d}} e^{i \omega \cdot x} \hat{f}(\omega) d \omega=\int_{\mathbb{R}^{d}} e^{i(\omega \cdot x+\beta(\omega))}|\hat{f}(\omega)| d \omega 
# $$
# 
# Since $f(x)$ is real-valued, it implies that, for $x$ 
# 
# $$
#     \begin{aligned}
# f(x) &=\operatorname{Re} \int_{\mathbb{R}^{d}} e^{i \omega \cdot x} \hat{f}(\omega) d \omega \\
# &=\operatorname{Re} \int_{\mathbb{R}^{d}} e^{i \omega \cdot x} e^{i \beta(\omega)}|\hat{f}(\omega)| d \omega \\
# &=\int_{\mathbb{R}^{d}} \cos (\omega \cdot x+\beta(\omega))|\hat{f}(\omega)| d \omega
# \end{aligned}
# $$
# 
# Then we have
# 
# $$
#     f(x)=\int_{\mathbb{R}^{d}} k(x, \omega) d \omega
# $$
# 
# with
# 
# $$
#     k(x, \omega)=\cos (\omega \cdot x+\beta(\omega))|\hat{f}(\omega)|
# $$
# 
# and 
# 
# $$
#     |k(x, \omega)| \leq|\hat{f}(\omega)|=\rho(\omega)
# $$
# 
# ```{prf:theorem}
# There exist $\omega_{i} \in \mathbb{R}^{d}$, s.t., $G=\mathbb{R}$ and
# 
# $$
#     \int_{\Omega}\left(f(x)-f_{n}(x)\right)^{2} \leq \frac{1}{n} \int_{\mathbb{R}^{d}}|\hat{f}(\omega)| d \omega
# $$
# 
# where
# 
# $$
#     f_{n}(x)=\frac{\|\hat{f}\|_{L^{1}}}{n} \sum_{i=1}^{n} \frac{\cos \left(\omega_{i}^{*} \cdot x+\beta_{i}^{*}\right)}{\rho\left(\omega_{i}^{*}\right)}
# $$
# 
# Note that
# 
# $$
#     f_{n}=\sum_{i=1}^{n} \frac{\cos \left(\omega_{i}^{*} \cdot x+\beta_{i}^{*}\right)}{\rho\left(\omega_{i}^{*}\right)} \in{ }_{n} \mathrm{~N}(\sigma, n)
# $$
# 
# with 
# 
# $$
#     \sigma(t)=\cos (t)
# $$
# 
# ```
# 
# ### Double Fourier representation
# 
# Assume that $\sigma$ is a locally Riemann integrable function and
# $\sigma \in L^{1}(\mathbb{R})$ and thus the Fourier transform of
# $\sigma$ is well-defined and continuous. Since $\sigma$ is non-zero and
# 
# $$
#     \hat{\sigma}(\omega)=\frac{1}{2 \pi} \int_{\mathbb{R}} \sigma(t) e^{-i \omega t} d t
# $$
# 
# this implies that $\hat{\sigma}(a) \neq 0$ for some $a \neq 0$. Via a
# change of variables $t=w \cdot x+b$ and $d t=d b$, this means that for
# all $x$ and $\omega$, we have 
# 
# $$
#     \begin{aligned}
# 0 \neq \hat{\sigma}(a) &=\frac{1}{2 \pi} \int_{\mathbb{R}} \sigma(\omega \cdot x+b) e^{-i a(\omega \cdot x+b)} d b \\
# &=e^{-i a \omega \cdot x} \frac{1}{2 \pi} \int_{\mathbb{R}} \sigma(\omega \cdot x+b) e^{-i a+b} d b
# \end{aligned}
# $$
# 
# and so
# 
# $$
#     e^{i a \omega \cdot x}=\frac{1}{2 \pi \hat{\sigma}(a)} \int_{\mathbb{R}} \sigma(\omega \cdot x+b) e^{-i a b} d b
# $$
# 
# Likewise, since the growth condition also implies that
# $\sigma^{(k)} \in L^{1}$, we can differentiate the above expression
# under the integral with respect to $x$.
# 
# This allows us to write the Fourier mode $e^{i a \omega \cdot x}$ as an
# integral of neuron output functions. We substitute this into the Fourier
# representation of $f$ (note that the assumption we make implies that
# $\hat{f} \in L^{1}$ so this is rigorously justified for a.e. $x$ ) to
# get
# 
# $f(x)=\int_{\mathbb{R}^{d}} e^{i \omega \cdot x} \hat{f}(\omega) d \omega=\int_{\mathbb{R}^{d}} \int_{\mathbb{R}} \frac{1}{2 \pi \hat{\sigma}(a)} \sigma\left(a^{-1} \omega \cdot x+b\right) \hat{f}(\omega) e^{-i a b} d b d \omega=\int_{\mathbb{R}^{d} \times \mathbb{R}} k(x, \theta) d \theta$
# where $\theta=(\omega, b)$ and
# 
# $$
#     k(x, \theta)=\frac{1}{2 \pi \hat{\sigma}(a)} \sigma\left(a^{-1} \omega \cdot x+b\right) \hat{f}(\omega) e^{-i a b}
# $$
# 
# Thus we have $$\begin{aligned}
# |k(x, \theta)| & \leq \frac{1}{2 \pi|\hat{\sigma}(a)|} \max _{x \in \Omega}\left|\sigma\left(a^{-1} \omega \cdot x+b\right) \| \hat{f}(\omega)\right| \\
# & \leq h(\omega, b)|\hat{f}(\omega)|=\rho(\theta)
# \end{aligned}$$ where
# 
# $$
#     h(\omega, b)=\max _{x \in \Omega}\left|\sigma\left(a^{-1} \omega \cdot x+b\right)\right|
# $$
# 
# if we ignore the coefficient. Thus, following the discussion in last
# section, the next step is to analyze $h(\omega, b)$ which we will
# discuss in the next section. Before that, let us introduce a special
# case of the above representation once the activation function is
# periodic.

# In[ ]:




