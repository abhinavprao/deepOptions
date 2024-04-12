from typing import Callable
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

ASSET_DIR = './assets'
key = jax.random.PRNGKey(0)


class FourierFeatures(eqx.Module):
    """
    A layer that outputs fourier features for a coordinate neural network.
    """
    B: jax.Array
    input_size: int
    mapping_size: int
    def __init__(self, input_size, mapping_size, key, scale = 1.):
        self.input_size = 1 if input_size=='scalar' else input_size
        self.mapping_size = mapping_size
        self.B = jax.random.normal(key, (self.input_size, mapping_size))*scale

    def __call__(self, x):
        x = jnp.atleast_1d(x)
        y = (2.*jnp.pi*x) @ self.B
        return jnp.concatenate([jnp.sin(y),jnp.cos(y)], axis=-1)


class MLPWithFourierFeatures(eqx.Module):
    """
    A neural net that calculates fourier features of the inputs and
    then applies the usual affine transformations of an MLP
    """
    input_size: int
    output_size: int
    mapping_size: int
    fourier_features: eqx.Module
    mlp: eqx.Module
    activation: Callable

    def __init__(self, input_size: int, output_size: int, mapping_size: int,
                 width_size: int , depth: int , key : jax.random.PRNGKey,
                 activation: Callable = jax.nn.tanh):
        self.input_size = input_size
        self.output_size = output_size
        self.mapping_size = mapping_size
        self.activation = activation

        key, fourier_key, mlp_key = jax.random.split(key, 3)
        self.fourier_features = FourierFeatures(input_size, mapping_size,
                                                fourier_key)
        self.mlp = eqx.nn.MLP(in_size = mapping_size*2, out_size = output_size,
                              width_size = width_size, depth =  depth,
                              activation = self.activation, key = mlp_key)

    def __call__(self, x):
        y = self.fourier_features(x)
        return self.mlp(y)
    

def plot_10_Cs(Cs, ts, Ss, sigmas, rs, name = 'evaluations'):
    '''Plots evaluations on 10 different points'''
    fig, axs = plt.subplots(5, 2, figsize=(10, 20))
    plt.suptitle("Paramteric Solution to Black Scholes equation", 
                 fontsize=18, y=1.01)

    for i, ax in enumerate(axs.flat):
        Cs_slice = Cs[:, :, i, i]
        T_plot, S_plot = jnp.meshgrid(ts, Ss)
        ax.axis('off')
        ax = fig.add_subplot(5, 2, i+1, projection='3d')
        surf = ax.plot_surface(T_plot, S_plot, Cs_slice.T, cmap='viridis')

        ax.set_xlabel('$t$')
        ax.set_ylabel('$S$')
        ax.set_zlabel('C(t, S)')
        ax.set_title(f'For $\sigma$={round(sigmas[i].item(),2)}'
                     f' and r={round(rs[i].item(), 2)}')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR,name), dpi=300, bbox_inches='tight')
