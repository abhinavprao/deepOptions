from functools import partial
import operator
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from utils import MLPWithFourierFeatures, plot_10_Cs
ASSET_DIR = './assets'

########################### SETUP AND HYPERPARAMS #############################

T = 1.                         # maturity
K = 2.5                        # strike price
S_lim = [0., 2.*K]             # stock price range
sigma_lim = [0, 0.5]           # volatility range
r_lim = [0, 0.1]               # risk free interest rate range

latent_size = 64               # latent size of deeponet
mapping_size_b = 50            # number of fourier frequencies for branch net
width_size_b = 256             # number of hidden neurons for branch net   
depth_b = 3                    # number of hidden layers for branch net
activation_b = jax.nn.tanh     # activation for branch net (doubly differential)
mapping_size_t = 50            # number of fourier frequencies for trunk net
width_size_t = 256             # number of hidden neurons for trunk net 
depth_t = 3                    # number of hidden layers for trunk net
activation_t = jax.nn.tanh     # activation for trunk net (doubly differential)

n_t = 32                       # batch size of time points
n_S = 32                       # batch size of stock price points
n_sigma = 8                    # batch size of volatility
n_r = 8                        # batch size of risk free interest rates
lr = 1e-4                      # learning rate
max_iter = 100000              # number of training iterations
update_every = 1000            # NTK normalization update rate
alpha = 0.2                    # NTK normalization update moving average agg
print_every = 100              # print rate


############################### ARCHITECTURE ##################################
key = jax.random.PRNGKey(0)
key, t_key, b_key= jax.random.split(key, 3)
# in equinox models are pytrees!
deeponet = {
    'Tr': MLPWithFourierFeatures(input_size = 2, 
                                 output_size = latent_size,
                                 mapping_size = mapping_size_t,  
                                 width_size=width_size_t, 
                                 depth=depth_t,
                                 key = t_key),
    'Br': MLPWithFourierFeatures(input_size = 2, 
                                 output_size = latent_size,
                                 mapping_size = mapping_size_b, 
                                 width_size=width_size_b, 
                                 depth=depth_b,
                                 key = b_key)}

#evaluate
def C(model, t, S, sigma, r):
    '''evaluates model for a single value of t, S, sigma, r'''
    Tr, Br = model['Tr'], model['Br']
    out = jnp.dot(Tr((t, S)), Br((sigma, r)))
    return S * (S_lim[1] - S) * out + S * ((S_lim[1] - K) / S_lim[1])#encoded BC

# gradients and parallel evaluation
C_t = jax.grad(C, argnums=1)
C_S = jax.grad(C, argnums=2)
C_SS = jax.grad(jax.grad(C, argnums=2), argnums=2)

def evaluate(model, ts, Ss, sigmas, rs):
    '''evaluates model for a multiple values of t, S, sigma, r'''
    v_C = jax.vmap(jax.vmap(jax.vmap(jax.vmap(C, 
            in_axes = (None, None, None, None, 0)), 
            in_axes = (None, None, None, 0, None)), 
            in_axes = (None, None, 0, None, None)), 
            in_axes = (None, 0, None, None, None))
    return v_C(model, ts, Ss, sigmas, rs)


############################## LOSS FUNCTION ##################################

@partial(jax.vmap, in_axes = (None, None, None, None, 0))
@partial(jax.vmap, in_axes = (None, None, None, 0, None))
@partial(jax.vmap, in_axes = (None, None, 0, None, None))
@partial(jax.vmap, in_axes = (None, 0, None, None, None))
def residual(model, t, S, sigma, r):
    '''evaluates squared PDE residual 
    for multiple values of t, S, sigma, r'''
    c = C(model, t, S, sigma, r)
    c_t = C_t(model, t, S, sigma, r)
    c_S = C_S(model, t, S, sigma, r)
    c_SS = C_SS(model, t, S, sigma, r)
    res = c_t + 0.5 * (sigma**2) * c_SS + r * c_S - r * c
    return jnp.square(res)

def net_residual(model, ts, Ss, sigmas, rs):
    '''evaluates mean squared PDE residual'''
    residuals = residual(model, ts, Ss, sigmas, rs)
    return jnp.mean(residuals)

def norm_grad_res(model, ts, Ss, sigmas, rs):
    '''evaluates normed gradient of PDE residual 
    for the purpose of normalising the loss'''
    res_grads = eqx.filter_grad(net_residual)(model, ts, Ss, sigmas, rs)
    squared_grads = jax.tree_util.tree_map(
        lambda x: jnp.square(jnp.linalg.norm(x)), res_grads)
    squared_norm = jax.tree.reduce(operator.add, squared_grads)
    return jnp.sqrt(squared_norm)

@partial(jax.vmap, in_axes = (None, None, None, 0))
@partial(jax.vmap, in_axes = (None, None, 0, None))
@partial(jax.vmap, in_axes = (None, 0, None, None))
def final_value_loss(model, S, sigma, r):
    '''evaluates squared final value residual 
    for multiple values of t, S, sigma, r'''
    c = C(model, T, S, sigma, r)
    val = jax.lax.cond(S>=K, lambda x: S - K, lambda x : jnp.array(0.), None)
    return jnp.square(c-val)

def net_fvl(model, Ss, sigmas, rs):
    '''evaluates mean squared final value residual'''
    fvls = final_value_loss(model, Ss, sigmas, rs)
    return jnp.mean(fvls)

def norm_grad_fvl(model, Ss, sigmas, rs):
    '''evaluates normed gradient of final value residual 
    for the purpose of normalising the loss'''
    fvl_grads = eqx.filter_grad(net_fvl)(model, Ss, sigmas, rs)
    squared_grads = jax.tree_util.tree_map(
        lambda x: jnp.square(jnp.linalg.norm(x)), 
        fvl_grads)
    squared_norm = jax.tree.reduce(operator.add, squared_grads)
    return jnp.sqrt(squared_norm)

def find_norm_weights(model, ts, Ss, sigmas, rs):
    g_res = jax.lax.stop_gradient(norm_grad_res(model, ts, Ss, sigmas, rs))
    g_fvl = jax.lax.stop_gradient(norm_grad_fvl(model, Ss, sigmas, rs))
    w_res = (g_res + g_fvl) / g_res
    w_fvl = (g_res + g_fvl) / g_fvl
    return w_res, w_fvl

def net_loss(model, ts, Ss, sigmas, rs, w_res, w_fvl):
    '''evaluates normalised squared loss'''
    res = net_residual(model, ts, Ss, sigmas, rs)
    fvl = net_fvl(model, Ss, sigmas, rs)
    return (w_res * res) + (w_fvl * fvl)

@eqx.filter_jit
def make_step(model, ts, Ss, sigmas, rs, w_res, w_fvl, optim, opt_state):
    loss, grads = eqx.filter_value_and_grad(net_loss)(model, ts, Ss, sigmas, rs, 
                                                      w_res, w_fvl)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


################################# TRAINING ###################################

if __name__ == '__main__':

	optim = optax.adam(lr)
	opt_state = optim.init(eqx.filter(deeponet, eqx.is_array))
	losses = []
	w_res, w_fvl = jnp.array(1.), jnp.array(1.)

	for i in range(max_iter):
		key, t_key, S_key, sigma_key, r_key = jax.random.split(key, 5)
		ts = jax.random.uniform(t_key, (n_t,), minval=0.0, maxval=T)
		Ss = jax.random.uniform(S_key, (n_S,), minval=S_lim[0], maxval=S_lim[1])
		sigmas = jax.random.uniform(sigma_key, (n_sigma,), minval=sigma_lim[0], 
		                            maxval=sigma_lim[1])
		rs = jax.random.uniform(r_key, (n_r,), minval=r_lim[0], maxval=r_lim[1]) 
		
		deeponet, opt_state, loss = make_step(deeponet,ts, Ss, sigmas, rs, 
		                                      w_res, w_fvl, optim, opt_state)
		
		losses.append(loss.item())

		if (i+1) % update_every == 0:
		    _w_res, _w_fvl = find_norm_weights(deeponet, ts, Ss, sigmas, rs)
		    w_res = alpha * _w_res + (1 - alpha) * w_res
		    w_fvl = alpha * _w_fvl + (1 - alpha) * w_fvl

		if (i+1) % print_every == 0:
		    print(f' iter = {i+1}, loss = {round(loss.item(),4)},'
		        f' weight_residual = {round(w_res.item(), 4)},'
		        f' weight_fvl={round(w_fvl.item(), 4)}')

	fig, ax = plt.subplots()
	ax.plot(losses)
	ax.set(title='loss v iterations', xlabel = 'iterations', ylabel = 'loss')
	fig.savefig(os.path.join(ASSET_DIR,'losses'), dpi=300, bbox_inches='tight')


	eqx.tree_serialise_leaves(os.path.join(ASSET_DIR, 'deeponet.eqx'), deeponet)

################################# EVALUATION ##################################

	ts = jnp.linspace(0, T, 100)
	Ss = jnp.linspace(S_lim[0], S_lim[1], 100)
	sigmas = jnp.linspace(sigma_lim[0], sigma_lim[1], 10)
	rs = jnp.linspace(r_lim[0], r_lim[1], 10)
	Cs = evaluate(deeponet, ts, Ss, sigmas, rs)
	plot_10_Cs(Cs, ts, Ss, sigmas, rs, name='evaluations')
    
