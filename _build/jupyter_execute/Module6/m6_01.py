#!/usr/bin/env python
# coding: utf-8

# # MgNet: a special CNN based on multigrid method 

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_luk1i2xr&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_s9zzktyl",width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2022/03/F01MgNet.pdf)

# 1D and 2D Finite Element and Multigrid
# -------------------------------------
# 
# 1D and 2D Comparison for Finite Element and Multigrid
# 
# ![image](images/img1.png)
# 
# Basic multigrid components
# 
# ![image](images/img2.png)
# 
# ### Multigrid algorithm for $A * \mu=f$
# 
# ```{prf:algorithm} A multigrid algorithm $\mu=\operatorname{MG} 1\left(f ; \mu^{0} ; J, v_{1}, \cdots, v_{J}\right)$
# :label: alg61_1
# Set up:
# 
# $$
#     f^{1}=f, \quad \mu^{1}=\mu^{0}
# $$
# 
# Smoothing and restriction from fine to coarse level (nested)
# 
# **For** $\ell=1: J$ do
# 
# **For** $i=1: v_{\ell}$ do
# 
# $$
#     \mu^{\ell} \leftarrow \mu^{\ell}+S^{\ell} *\left(f^{\ell}-A_{\ell} * \mu^{\ell}\right)
# $$
# 
# **EndFor**
# 
# Form restricted residual and set initial guess:
# 
# $$
#     \mu^{\ell+1} \leftarrow \Pi_{\ell}^{\ell+1} \mu^{\ell}, \quad f^{\ell+1} \leftarrow R *_{2}\left(f^{\ell}-A_{\ell} * \mu^{\ell}\right)+A_{\ell+1} * \mu^{\ell+1}
# $$
# 
# **EndFor**
# 
# Prolongation and restriction from coarse to fine level
# 
# **For** $\ell=J-1: 1$ do
# 
# $$
#     \mu^{\ell} \leftarrow \mu^{\ell}+R *_{2}^{\top}\left(\mu^{\ell+1}-\Pi_{\ell}^{\ell+1} \mu^{\ell}\right)
# $$
# 
# **EndFor**
# 
# $$
#     \mu \leftarrow \mu^{1}
# $$ 
# 
# ```
# 
# 
# ```{admonition} Remark
# The above multigrid method
# for the linear problem $A * \mu=b$ is independent of the choice of the
# interpolation operation
# $\Pi_{\ell}^{\ell+1}: \mathbb{R}^{n_{\ell} \times n_{\ell}} \mapsto \mathbb{R}^{n_{\ell+1} \times n_{\ell+1}}$
# and in particular, we could take $\Pi_{\ell}^{\ell+1}:=0$. But such an
# operation is critical for nonlinear problems.
# 
# ### MgNet
# 
# ```{prf:algorithm} $\mu^{J}=\operatorname{MgNet} 1\left(f ; \mu^{0} ; J, v_{1}, \cdots, v_{J}\right)$
# :label: alg61_2
# Set up:
# 
# $$
#     \qquad f^{1}=\theta * f, \quad \mu^{1}=\mu^{0} 
# $$
# 
# Smoothing and restriction from fine to coarse level (nested)
# 
# **For** $\ell=1: J$ do 
# 
# $\quad$
# 
# **For** $i=1: v_{\ell}$ do
# 
# $$
#     \quad \mu^{\ell} \leftarrow \mu^{\ell}+\sigma \circ S^{\ell} * \sigma \circ\left(f^{\ell}-A_{\ell} * \mu^{\ell}\right)
# $$
# 
# **EndFor**
# 
# Form restricted residual and set initial guess:
# 
# $$
#     \quad \mu^{\ell+1} \leftarrow \Pi_{\ell}^{\ell+1} \mu^{\ell}, \quad f^{\ell+1} \leftarrow R *_{2}\left(f^{\ell}-A_{\ell} * \mu^{\ell}\right)+A_{\ell+1} * \mu^{\ell+1}
# $$
# 
# **EndFor**
# ```
# 

# In[ ]:




