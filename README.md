# Parametric problem over Black Scholes Merton using DeepOnets


## Premise of Black Scholes Merton Equation:

Consider an stock valued at $S$. For a call option issued over this stock with strike price $K$ and maturation time $T$, we can say the apporpriate price for the call option $C(t, S)$ is given by the Black Scholes Merton equation:

$$
\dfrac{\partial C}{\partial t} + \dfrac{1}{2}\sigma^2 S^2 \dfrac{\partial^2 C}{\partial S^2} + rS\dfrac{\partial C}{\partial S} -rC = 0
$$
Where $\sigma$ is the volatility (a constant) and $r$ is the riskfree interest rate (say, the treasury bond interest rate).<br>

For European type call options (sells at maturity), we can write the boundary conditions and final value at $T$ as:
$$
\begin{align*}
   &C(t, 0) = 0 &&\text{for all $t \geq 0$}\\
   &C(t, S) \to S - K &&\text{for all $t \geq 0$ as $S \to \infty$}\\
   &C(T, S) = \max\{S - K, 0\}
\end{align*}
$$

## Why use neural networks?
We can solve this PDE, using  neural networks (PINNs), but ultimately PINNs do not offer any advantage over numerical methods for this case. They are relatively inaccurate, take a long time to train, and are riddled with edge case pathologies. 

But if we consider the parametric case, i.e. the parameters $\sigma$ and $r$ are independent random variables themselves, then the problem becomes much more tedious for numerical methods. There are different ways to view this problem - 

1. Since the parameters of the PDE are random variables, it is a case of stochastic differential equation, with a random field as solution. We can approximate using PINNs and use MCMC to characterize the posterior.
2. The PDE can be viewed as an operator over the field function and taking the form $\mathcal{B}$ that maps  $\mathcal{B} :\sigma, r \rightarrow C_{\delta, r}(t, S)$

We choose the latter problem formulation and we can then treat the operator using neural networks $\Beta$ (unstacked branch network) and $\Psi$ (trunk network):

$$
B_w(t, S ;\sigma, r) = \left(\sum_{i=1}^n \Beta_{x,i}(\delta)\Psi_{x,i}(t, S) \;, \;\sum_{i=1}^n \Beta_{x,i}(\sigma, r)\Psi_{x,i}(t, S)\right)
\sim C_{(\sigma, r)}(t, S)$$

with the loss function spelled out as,

$$
\mathcal{L}(w) = \int_0^{0.1} \int_0^1  \int_0^\infty \int_0^T \left\{\left(\dfrac{\partial}{\partial t} + \dfrac{1}{2}\sigma^2 S^2 \dfrac{\partial^2}{\partial S^2} + rS\dfrac{\partial}{\partial S} -r \cdot() \right)\left[C_{(\sigma, r)}(t, S)\right] \right\}dt dS d\sigma dr
$$

$$
\mathcal{L}(w) = \int_0^{r_{max}} \int_0^{\sigma_{max}}  \int_0^\infty \int_0^T  B_w(t, S ;\sigma, r) dt dS d\sigma dr
$$