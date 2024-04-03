# Pricing Options using Deep Operator Learing

A demonstration for pricing call options using neural networks.

## The Black Scholes Merton Equation:

Consider an security valued at $S$. For a call option superjacent to this security with strike price $K$ and maturation time $T$, the apporpriate price for the call option $C(t, S) \;\; \forall t < T \; \; \& \;\; S \in [0,\infty)$ is given by the Black Scholes Merton equation:

$$
\dfrac{\partial C}{\partial t} + \dfrac{1}{2}\sigma^2 S^2 \dfrac{\partial^2 C}{\partial S^2} + rS\dfrac{\partial C}{\partial S} -rC = 0
$$
Where $\sigma$ is the volatility of the security and $r$ is the riskfree interest rate, say, the treasury bond interest rate. Also based on the nature of the call option contract, we can write the boundary conditions and final value condition at $T$ as:
$$
\begin{align*}
   &C(t, 0) = 0 &&\text{for all $t \geq 0$}\\
   &C(t, S) \to S - K &&\text{for all $t \geq 0$ as $S \to \infty$}\\
   &C(T, S) = \max\{S - K, 0\}
\end{align*}
$$

**Intuition:** If one chooses the perfectly conservative move; making a position on a security (long or short), while hedging the position against risk with the appropriately balancing option (put or call), then one's expected profit rate should be the same as the risk-free interest rate in the market. This push and pull between the option price and stock price is illustrated by the above differential equation.

**Note:** This formulation makes a bunch of assumptions:
1. Security price is a GRF, with constant volatility. The security trading is pure, it does not pay dividends - its price is set purely by demand-supply.
2. The riskfree rate remains constant as well.
3. The option is only sold at maturity (European, not American options).

## Why (and why not) use neural networks?
The above equation is a partial differential equation (PDE). We can solve this PDE, using neural networks (PINNs), but ultimately PINNs do not offer any advantage over numerical methods for this case. They are relatively inaccurate, take a long time to train, and are riddled with boundary pathologies.

But if we consider the parametric problem statement, i.e. the parameters $\sigma$ and $r$ are independent random variables (not random fields, yet) themselves, then the problem becomes much more tedious and scales poorly with numerical methods. But even this formulation can be solved easily with PINNs with a somewhat ad-hoc formulation, but there is a more appropriate way to do this

There are different ways to view this problem - 

1. Since the parameters of the PDE are random variables, it is a case of simply a parametric stochastic differential equation, with a stochastic process as solution. We can approximate using PINNs and use MCMC-type methods to characterize the posterior.
2. More appropriately and simply, the PDE can be viewed as an operator over the field function, of the form $\mathcal{K} : \mathcal{\Psi}(\sigma, r) \rightarrow C_{\delta, r}$, mapping the parameters to a functions in the $\mathcal{L}^2$ space.

## What is an Operator, and how do we learn it?

An operator, simply put, is a maps a function to a function, for example $\frac{d(\cdot)}{dx}$. Symbolically $G: V \rightarrow C$ such that $f(x) := G(u)(x)\;\;\;$ for $f\in C$ and $u \in V$.

Turns out, neural networks are [universal approximators of nonlinear operators](https://ieeexplore.ieee.org/document/392253) just like they are universal approximators of functions. But how can we input functions into a neural networks? This is a bit unintuitive at first, and once you write it down, it might seem a bit ad-hoc, but fundamentally there is no difference between operator learning and function learning from a neural network view-point. We essentially just pose the operator learning as a function learning problem. We follow the formulation and conventions of [Lu et al.](https://arxiv.org/abs/1910.03193)

Let us just consider scalar valued functions for now, multiple inputs - single output. Then 

We choose the latter problem formulation and we treat the operator on a collocation of points following the formulation of Karniadak:

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