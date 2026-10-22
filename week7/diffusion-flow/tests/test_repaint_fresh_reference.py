"""Coordinator-authored independent regression of the formal fresh RNG path.

Equations use NumPy float64 and the declared key protocol; the production
call does NOT provide noise_stream/xT, so it exercises the compiled core.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.repaint import repaint_sample


def local_predictor(weights, x, index):
    return .09*x+.11*jnp.mean(x,axis=(-2,-1),keepdims=True)+.07*index+weights['gain']


def numpy_path(seed, observed, mask, abar, wrong_variance=False):
    init, carry = jax.random.split(jax.random.PRNGKey(seed))
    x = np.asarray(jax.random.normal(init,observed.shape,dtype=jnp.float32)).astype(np.float64)
    rows = []
    for t,s,retry in [(4,2,False),(4,2,True),(2,0,False)]:
        kf,kxi,keta,carry = jax.random.split(carry,4)
        f,xi,eta = [np.asarray(jax.random.normal(k,observed.shape,dtype=jnp.float32)).astype(np.float64) for k in [kf,kxi,keta]]
        at,ass = float(abar[t]),float(abar[s]); a=at/ass
        if retry:
            x = np.sqrt(a)*x+np.sqrt(1-a)*f
        eps = .09*x+.11*x.mean(axis=(-2,-1),keepdims=True)+.07*(t-1)+.025
        clean = np.clip((x-np.sqrt(1-at)*eps)/np.sqrt(at),-1,1)
        mean = np.sqrt(ass)*(1-a)/(1-at)*clean+np.sqrt(a)*(1-ass)/(1-at)*x
        variance = (1-a) if wrong_variance else (1-ass)*(1-a)/(1-at)
        proposal = mean+np.sqrt(variance)*xi
        known = np.sqrt(ass)*observed+np.sqrt(1-ass)*eta
        x = np.where(mask==1,known,proposal)
        rows.append(x.copy())
    return np.stack(rows)


@pytest.mark.parametrize('seed',[8642,8643])
def test_fresh_compiled_path_matches_independent_coarse_posterior(seed):
    shape=(2,1,2,3)
    mask=np.ones(shape,np.float32); mask[...,0,1:]=0; mask[...,1,0]=0
    observed=np.linspace(-.8,.8,np.prod(shape),dtype=np.float32).reshape(shape)*mask
    betas=np.array([.1,.2,.3,.4],np.float32)
    abar=np.concatenate([np.ones(1,np.float32),np.cumprod(1-betas,dtype=np.float32)])
    actual,states=repaint_sample({'gain':jnp.float32(.025)},jnp.array(observed),jnp.array(mask),jnp.array(betas),jnp.array(abar),[4,2,0],attempts=2,key=jax.random.PRNGKey(seed),predict=local_predictor,record_states=True)
    expected=numpy_path(seed,observed,mask,abar)
    np.testing.assert_allclose(np.asarray(states),expected,atol=2e-6,rtol=2e-6)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(states)[-1])
    np.testing.assert_array_equal(np.asarray(actual)[mask==1],observed[mask==1])
    # A coarse-DDPM-variance mutant in the formal branch would fail this
    # regression, even if the separate explicit-stream branch stayed correct.
    wrong=numpy_path(seed,observed,mask,abar,wrong_variance=True)
    assert np.max(np.abs(wrong-expected))>.01
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(states),wrong,atol=2e-6,rtol=2e-6)
