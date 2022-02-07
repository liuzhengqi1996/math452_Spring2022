#!/usr/bin/env python
# coding: utf-8

# # Finite element method

# In[1]:


from IPython.display import IFrame

IFrame(src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_mpdchgne&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_4q3v8u79" ,width='800', height='500')


# ## Download the lecture notes here: [Notes](https://sites.psu.edu/math452/files/2022/01/C03_-Finite-Element-Methods.pdf)

# Finite element method is a classic numerical method for solving partial
# differential equations. In this chapter, we will give a brief
# introduction to this method. discuss its basic properties and error
# estimates. In later chapters, we will show that the neural network
# functions can be viewed as an extension of finite element function. In
# this chapter, we discuss the classical linear finite element spaces, the
# error estimate of the finite element method and adaptivity method to
# improve the approximation. For shape-regular mesh, we will establish
# both the upper and lower bound of the approximation error.
# 
# ## Linear finite element spaces 
# ============================
# 
# In this section, we introduce linear finite element spaces. We will walk
# through the basic setup, and derive some error estimates.
# 
# ### Triangulations
# --------------
# 
# Given a bounded polyhedral domain $\Omega \subset \mathbb {R}^d$, a
# geometric triangulation (also called mesh or grid)
# $\mathcal T_h=\{\tau\}$ of $ \Omega $ is a set of $d$-simplices such that
# 
# 1.  $\overline \Omega=\cup \tau$, where $ \overline \Omega$ denotes the
#     closure of $\Omega$.
# 
# 2.  if $\tau_1$ and $\tau_2$ are distinct elements in $\mathcal T_h$
#     then
#     $\stackrel{\circ}{\tau _1}\cap \stackrel{\circ}{\tau _2} = \varnothing$,
#     where $\stackrel{\circ}{\tau _i}$ denotes the interior of
#     $\tau_i, i=1,2$ .
# 
# Examples of triangulations for $\Omega=(0,1)$ ($d=1$) and for
# $\Omega=(0,1)^2$ ($d=2$) are shown in 
# 
# ```{figure} ./images/img1.png
# :height: 150px
# :name: 1dgrid
# 1D uniform grid
# ```
# 
# and
# 
# ```{figure} ./images/img2.png
# :height: 150px
# :name: 2dgrid
# 2D uniform grid
# ```
# 
# 
# Given $\Omega=(0,1)$, we consider the mesh: 
# 
# $$
#     0=x_0<x_1<\cdots<x_{n+1}=1, \quad x_i=\frac{i}{n+1},\quad (i=0,\cdots,n+1)
# $$ (partitionyx)
# 
# which is the 1D uniform grid shown in figure {numref}`1dgrid`
# 
# 
# Denote
# 
# $$
#     h_\tau=\mbox{ diam} (\tau)\quad  \hbox{(diameter of the smallest sphere containing $\overline{\tau}$)},
# $$
# 
# and
# 
# $$
#     h=\max_{\tau\in\mathcal T_h} h_\tau;\quad
#     \underline{h}=\min_{\tau\in\mathcal T_h} h_\tau.
# $$ 
# 
# A set of triangulations $\mathscr T$ is called shape regular if there
# exists a constant $c_0$ such that
# 
# $$
#     \max _{\tau \in \mathcal T_h} \frac{h_{\tau}^d}{|\tau|}\leq c_0, \quad \forall \, \mathcal T_h\in \mathscr T
# $$ (shape)
# 
# where $|\tau|$ is the measure of $\tau$ in $ R^d$.
# This assumption can also be represented as 
# 
# $$
#     \max_{\tau\in\mathcal T_h}\frac{h_\tau}{\rho_\tau}\le\sigma_1,\quad \forall \, \mathcal T_h\in
#     \mathscr T
# $$ (A3.1)
# 
# where $\rho_\tau$ denotes the radius of the ball inscribed
# in $\tau$. In two dimensions, it is equivalent to the minimal angle of
# each triange is bounded below uniformly in the shape regular class. 
# 
# In addition to {eq}`shape`, if 
# 
# $$
#   \frac{\max _{\tau \in \mathcal T_h}|\tau|}{\min _{\tau \in \mathcal T_h}|\tau|} \leq \rho,\quad \forall \, \mathcal T_h\in \mathscr T
# $$ (A3.2)
# 
# $\mathscr T$ is called quasi-uniform. For quasi-uniform grids,
# $h=\max _{\tau \in \mathcal T_h} h_{\tau}$, the mesh size of
# $\mathcal T_h$, is used to measure the approximation rate.
# 
# The assumption {eq}`A3.2` is a local assumption, as is meant by above definition, for
# $d=2$ for example, it assures that each triangle will not degenerate
# into a segment in the limiting case. On the other hand, the assumption
# is a global assumption, which says that the smallest mesh size is not
# too small compared with the largest mesh size of the same triangulation.
# By the definition, in a quasi-uniform triangulation, all the elements
# are about the same size asymptotically.
# 
# Let $ x_{i}=(x^1_{i}, \cdots, x^d_{i})^t, i=1,\cdots, d+1$ be $d+1$
# points in $ R^d$ which do not all lie in one hyper-plane. The
# convex hull of the $d+1$ points $ x_1, \cdots,  x_{d+1}$ (See
# Figure \[fig:barycentricCoor\])
# 
# $$
#     \tau :=\{ x=\sum _{i=1}^{d+1}\lambda _i x_i \, | \, 0\leq \lambda_i\leq 1, i=1:d+1, \sum _{i=1}^{d+1}\lambda _i=1 \}
# $$
# 
# is defined as a geometric $d$-simplex generated (or spanned) by
# the vertices $ x_1, \cdots,  x_{d+1}$. For example, a triangle is a
# $2$-simplex and a tetrahedron is a $3$-simplex. For an integer
# $0\leq m \leq d-1$, an $m$-dimensional face of $\tau$ is any $m$-simplex
# generated by $m+1$ of the vertices of $\tau$. Zero-dimenisonal faces are
# vertices and one-dimensional faces are called edges of $\tau$. The
# $(d-1)$-face opposite to the vertex $ x_i$ will be denoted by $F_i$.
# 
# ```{figure} ./images/img3.png
# :height: 150px
# :name: geo
# Geometric explanation of barycentric coordinates
# ```
# 
# 
# On the other hand, for any $ x\in \tau$, there exist unique numbers
# $\lambda _1,\cdots, \lambda _{d+1}$ satisfying
# $\displaystyle 0\leq \lambda_i\leq 1, i=1:d+1, \sum _{i=1}^{d+1}\lambda _i=1$
# such that $\displaystyle x=\sum _{i=1}^{d+1}\lambda _i x_i$, thus we can
# denote $\lambda _1,\cdots, \lambda _{d+1}$ as
# $\lambda _1( x),\cdots, \lambda _{d+1}( x)$. In fact, the numbers
# $\lambda _1( x),\cdots, \lambda _{d+1}( x)$ are called barycentric
# coordinates of $ x$ with respect to the $d+1$ points
# $ x_1, \cdots,  x_{d+1}$. There is a simple geometric meaning of the
# barycentric coordinates. Given a $ x\in
# \tau$, let $\tau _i( x)$ be the simplex with vertices $ x_i$ replaced by
# $ x$. Then it can be easily shown that 
# 
# $$
#     \lambda _i( x) = |\tau _i( x)|/|\tau|,
# $$ (eq:lambdasolution)
# 
# where $|\cdot|$ is the Lebesgure measure in $ R^d$, namely area in two dimensions and
# volume in three dimensions. Note that $\lambda
# _i( x)$ is affine function of $ x$ and vanishes on the face $F_i$. We
# list the four basic properties of barycentric coordinate below:
# 
# 1.  $0\leq \lambda_i( x)\leq 1$;
# 
# 2.  $\displaystyle\sum_{i=1}^{d+1} \lambda_i( x)=1$;
# 
# 3.  $\lambda_i( x)\in P_1(\tau)$, where $P_1(\tau)$ denotes the space of
#     polynomials of degree $1$ (linear) on $\tau\in \mathcal T_h$;
# 
# 4.  $\lambda_i( x_j)=\delta_{ij}=\begin{cases}
#     1, \quad &\text{if}  \quad  i=j\\
#     0, \quad &\text{if} \quad i\neq j
#     \end{cases}.$
# 
# ## Continuous linear finite element spaces
# ---------------------------------------
# 
# A conforming linear finite element function in a domain $\Omega\subset
# \mathbb R^d$ is a continuous function that is piecewise linear function
# with respect to a grid or mesh consisting of a union of simplices.
# 
# Given a shape regular triangulation $\mathcal T_h$ of $\Omega$, we
# define the continuous linear finite element space as 
# 
# $$
#     V_h:=\{v\,|\, v\in C(\overline \Omega), \,\hbox{ and }\, v|_{\tau}\in
# P_1(\tau), \forall \tau \in \mathcal T_h\}
# $$ (LinFE)
# 
# where $P_1(\tau)$ denotes
# the space of polynomials of degree $1$ (linear) on
# $\tau\in \mathcal T_h$. Whenever we need to deal with boundary
# conditions, we further define $V_{h,0}=V_h\cap H_0^1(\Omega)$.
# 
# We note here that the global continuity is also necessary in the
# definition of $V_h$ in the sense that if $u$ has a square interable
# gradient, that is $u\in H^1(\Omega)$, and $u$ is piecewise smooth, then
# $u$ is continuous.
# 
# We always use $n_h$ to denote the dimension of finite element spaces.
# For $V_h$, $n_h$ is the number of vertices of the triangulation
# $\mathcal T_h$ and for $V_{h,0}$, $n_h$ is the number of interior
# vertices.
# 
# #### Nodal basis functions and dual basis
# 
# For linear finite element spaces, we have the so called *a standard
# nodal basis functions* $\{\varphi
# _i,i=1,\cdots n_h\}$ such that $\varphi_i$ is piecewise linear (with
# respect to the triangulation) and $\varphi_i(x_j)=\delta_{i,j}$. Note
# that $\varphi _i|_\tau$ is the corresponding barycentrical coordinates
# of $x_i$. See Figure {numref}`nodal_basis` for an illustration in 2D.
# 
# ```{figure} ./images/img4.png
# :height: 150px
# :name: dual_basis
# Dual basis functions of Vh in 1D for nh = 5.
# ```
# 
# Let $(\varphi_i^*)_{i=1}^{n_h}$ be the dual basis of
# $(\varphi_i)_{i=1}^{n_h}$, namely 
# 
# $$
#     (\varphi_i^*, \varphi_j) =\delta_{i, j}, \quad i, j=1,\ldots, n_h
# $$ (eq1)
# 
# We notice that all the nodal basis functions $\{\varphi_i\}$ are locally
# supported, but their dual basis functions $\{\varphi_i^*\}$ are in
# general not locally supported (see Figure {numref}`dual_basis`). The nodal
# basis functions $\{\varphi_i\}$ are easily constructed in terms of
# barycentric coordinate functions. The dual basis $\{\varphi_i^*\}$ are
# only interesting for theoretical consideration and it is not necessary
# to know the actual constructions of these functions.
# 
# ```{figure} ./images/img5.png
# :height: 500px
# :name: nodal_basis
# Nodal basis functions in 1d and 2d
# ```
# Since $\{\varphi_i,i=1,\cdots n_h\}$ is a basis of $V_h$, therefore for
# any $v_h\in V_h$, we have the representation
# 
# $$
#     v_h(x)=\sum_{i=1}^{n_h}v_h(x_i)\varphi_i(x)
# $$
# 
# Let us see how our construction of continuous linear finite space and
# the nodal basis looks like in one spatial dimension. Associated with the
# mesh 
# 
# $$
#     \mathcal T_h=\{0=x_0<x_1<\ldots<x_{n_h}<x_{n_h+1}=1\}
# $$
# 
# by the definition given in {eq}`LinFE` and the definition
# $V_{h,0}=V_{h}\cap H_0^1(\Omega)$, we have 
# 
# $$
#     \begin{array}{ll}
# V_{h,0}=\{v:~\mbox{$v$ is continuous and piecewise linear}~ \mbox{w.~r.~t. $\mathcal T_h$, } v(0)=v(1)=0\}.
# \end{array}
# $$ 
# 
# A plot of a typical element of $V_{h,0}$ is shown in
# Figure {numref}`plot`
# 
# It is easily calculated (as we already mentioned), that the dimension of
# $V_{h,0}$ is equal to the number of internal vertices, and the nodal
# basis functions spanning $V_{h,0}$ (for $i=1,2,\cdots,n_h$) are (see
# also Fig {numref}`nodal_basis` 
# 
# $$
#     \varphi_i(x)=\left\{\begin{array}{cl}
#     \displaystyle \frac{x-x_{i-1}}{h}, & x\in[x_{i-1},x_i];\\
#     \displaystyle \frac{x_{i+1}-x}{h}, & x\in[x_{i},x_{i+1}];\\
#     0 &\mbox{elsewhere}.
#     \end{array}\right.
# $$ (1dbasis:function)
# 
# ```{figure} ./images/img6.png
# :height: 500px
# :name: plot
# Plot of a typical element from Vh;0 .
# ```
# 
# 

# #### Nodal value interpolant
# 
# For any continuous function $u$, we define its linear finite element
# interpolation, $(I_h u)(x)\in V_{h,0}$, as follows: 
# 
# $$
#     (I_h u)(x)= \sum_{i=1}^{n_h}u(x_i)\varphi_i(x)
# $$ (u-interp)
# 
# Usually, we also
# denote $(I_h u)(x)$ as $u_I(x)$. Using interpolation, we can obtain the
# following approximation property of linear finite element space.
# 
# ```{prf:theorem}
# :label: interp00
# Assume that $\mathcal T_h$ is quasi-uniform and $V_h$ is
# the linear finite element space associated with $\mathcal T_h$, then
# 
# $$
#     \inf_{v_h\in V_h} \|v-v_h\|+h |v-v_h|_{1}\leq h^2 |v|_2
#         \forall v\in H^2(\Omega).
# $$ (error0)
# ```
# ```{prf:proof} Proof
# Let us first prove Theorem {prf:ref}`interp00` for $d=1, 2, 3$. 
# Let $x=(x^1,\ldots, x^d)$ and
# $a_i=(a^1_{i}, \ldots, a^d_{i})$. Introducing the auxiliary functions
# 
# $$
#     g_i(t)=v(a_i(t)),\mbox{  with  }  a_i(t)=a_i+t(x-a_i)
# $$
# 
# we have
# 
# $$
#     g_i'(t)=(\nabla v)(a_i(t))\cdot (x-a_i)
# =\sum_{l=1}^d(\partial_lv)(a_i(t))(x^l-a_i^l)
# $$ 
# 
# and 
# 
# $$
#     g_i''(t)=\sum_{k,l=1}^d\partial^2_{kl}v)(a_i(t))(x^k-a_i^k)(x^l-a_i^l)
# $$ (gpp)
# 
# Note Taylor expansion 
# 
# $$
#     g_i(0)=g_i(1)-g_i'(1)+\int_0^1tg''_i(t)dt
# $$
# 
# 
# namely 
# 
# $$
#     v(a_i)=v(x)-(\nabla v)(x)\cdot (x-a_i)+\int_0^1tg''_i(t)dt
# $$ (Taylor_vi)
# 
# and note that
# 
# $$
#     (I_hv)(x)=\sum_{i=1}^{d+1}v(a_i)\lambda_i(x), \quad \sum_{i=1}^{d+1}\lambda_i(x)=1
# $$
# 
# and 
# 
# $$
#     \sum_{i=1}^{d+1}(x-a_i)\lambda_i(x)=0
# $$
# 
# It follows that
# 
# $$
#     (I_hv-v)(x)=\sum_{i=1}^{d+1}\lambda_i(x)\int_0^1tg''_i(t)dt
# $$ (Ihvv)
# 
# Using {eq}`gpp` and the trivial fact that $|x^l-a_i^l|\le h$, we obtain 
# 
# $$
#     \begin{aligned}
#     \|g''_i(t)\|_{L^2(\tau)}\le h^2
#     \sum_{k,l=1}^d\|(\partial^2_{kl}v)(a_i(t))\|_{L^2(\tau_i^t)}
#     \le h^2t^{-d/2}\sum_{k,l=1}^d\|\partial^2_{kl}v\|_{L^2(\tau)},\end{aligned}
# $$
# 
# where we have used the following change of variable
# 
# $$
#     y=a_i+t(x-a_i): \tau\mapsto \tau_i^t\subset\tau \mbox{ with } dy=t^ddx
# $$
# 
# Now taking the $L^2(\tau)$ norm on both hand of sides of {eq}`Ihvv` , we get
# 
# $$
#     \begin{aligned}
#     \|I_hv-v\|_{L^2(\tau)}
#     &\le& h^2\sum_{i=1}^{d+1}\max_{x\in\tau}|\lambda_i(x)|
#     \int_0^1t\|g''_i(t)\|_{L^2(\tau)}\;dt\\
#     &\le& (d+1)\int_0^1t^{-d/2}dt\;h^2\;
#     \sum_{k,l=1}^d\|\partial^2_{kl}v\|_{L^2(\tau)}\\
#     &\le&\frac{2(d+1)}{4-d}h^2
#     \sum_{k,l=1}^d\|\partial^2_{kl}v\|_{L^2(\tau)}\\
#     &\le&\frac{4d(d+1)}{4-d}h^2|v|_{H^2(\tau)}.\end{aligned}
# $$
# 
# Now we prove the $H^1$ error estimate. Notice that
# 
# $$
#     [\partial_{j}( I_{h} v - v)](x) = \sum_{i} (\partial_{j} \lambda_{i} )(x) \int_{0}^{1} t g''_{i}(t) dt + \sum_{i} \lambda_{i}(x) \partial_{j} \int_{0}^{1} t g''_{i}(t) dt
# $$
# 
# By {eq}`Taylor_vi`,
# 
# $$
#     \int_0^1tg''_i(t)dt = v(a_i) - v(x) + (\nabla v)(x)\cdot (x-a_i)
# $$
# 
# therefore, 
# 
# $$
#     \begin{aligned}
#     \partial_{j} \int_0^1tg''_i(t)dt \\
#     & = & - \partial_{j} v + (\nabla \partial_{j} v )(x) (x - a_{i}) + \nabla v \cdot e_{j} \\
#     & = & (\nabla \partial_{j} v )(x) (x - a_{i}).\end{aligned}
# $$
# 
# where $e_{j}$ is the $j$-th standard basis 
# 
# Noting that $\sum_{i} \lambda_{i}( \nabla \partial_{j} v )(x) (x - a_{i}) = 0$,
# we have
# 
# $$
#     [\partial_{j}( I_{h} v - v)](x) = \sum_{i} (\partial_{j} \lambda_{i} )(x) \int_{0}^{1} t g''_{i}(t) dt
# $$
# 
# Then the estimate for $|\nabla(I_hv-v)|_{L^2(\tau)}$ follows by a
# similar argument and the following obvious estimate
# 
# $$
#     |(\nabla\lambda_i)(x)|\leq\frac{1}{h}
# $$
# 
# On the proof of Theorem {prf:ref}`interp0` for $d\ge 4$, the above proof does
# not apply for $d \ge 4$. This is because when $d \ge 4$, the embedding
# relation between $H^{2}(\Omega) \hookrightarrow C(\bar{\Omega})$ is no longer
# true. Only continuous functions can have interpolations. In this case,
# one approach is to use the so-called Scott-Zhang interpolation
# ```
# 
# As a result of Theorem {prf:ref}`interp00` , we have
# 
# ```{figure} ./images/img8.png
# :height: 500px
# :name: Approximatio
# Approximation of finite element space.
# ```
# 
# 
# ```{prf:theorem}
# :label: interp0
# Let $V_N$ be linear finite element space on a quasi-uniform
# triangulation consisting of $N$ element. Then 
# 
# $$
#     \inf_{v_h\in V_N} \|v-v_h\|+N^{-{1\over d}} |v-v_h|_{1}\leq N^{-{2\over d}} |v|_2
#     \forall   v\in H^2(\Omega)
# $$
# 

# In[ ]:




