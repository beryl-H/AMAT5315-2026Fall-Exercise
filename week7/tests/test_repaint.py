"""Breaks caught: wrong posterior, clean intermediate pixels, RNG/index errors."""
import importlib
import numpy as np
import pytest
import jax
import jax.numpy as jnp


def api():
    try:return importlib.import_module('repaint')
    except ModuleNotFoundError:raise AssertionError('Part 3 conditional sampler is not implemented')


def test_grid_contains_exactly_298_attempts_and_one_final():
    t,s,c,again=api().transition_grid(100,3)
    assert len(t)==298
    np.testing.assert_array_equal(t[:3],[1000]*3)
    np.testing.assert_array_equal(c[:3],[0,1,2])
    assert (t[-1],s[-1],c[-1],again[-1])==(1,0,0,False)
    assert np.count_nonzero(s==0)==1
    assert (t>s).all()
    with pytest.raises(ValueError):api().transition_grid(1001,3)


def test_skipped_posterior_matches_scalar_reference_and_is_unclipped_after_merge():
    x=np.array([[[[.6],[.3]]]],np.float32)
    eps=np.array([[[[.1],[-.2]]]],np.float32)
    y=np.array([[[[.8],[0.]]]],np.float32);m=np.array([[[[True],[False]]]])
    z=np.full_like(x,.4);eta=np.full_like(x,2.)
    bt=.2;bs=.6;a=bt/bs
    clean=np.clip((x-np.sqrt(1-bt)*eps)/np.sqrt(bt),-1,1)
    mean=np.sqrt(bs)*(1-a)/(1-bt)*clean+np.sqrt(a)*(1-bs)/(1-bt)*x
    var=(1-bs)*(1-a)/(1-bt)
    expected=np.where(m,np.sqrt(bs)*y+np.sqrt(1-bs)*eta,mean+np.sqrt(var)*z)
    got=np.asarray(api().reverse_step(jnp.array(x),jnp.array(eps),jnp.array(y),jnp.array(m),jnp.float32(bt),jnp.float32(bs),jnp.array(z),jnp.array(eta)))
    np.testing.assert_allclose(got,expected,atol=2e-7)
    assert got[0,0,0,0]>1. # A post-merge clip would fail this.


def test_final_step_exact_observations_and_clipped_clean_hidden():
    x=jnp.array([[[[4.],[-4.]]]],dtype=jnp.float32)
    y=jnp.array([[[[.3],[0.]]]],dtype=jnp.float32)
    m=jnp.array([[[[True],[False]]]])
    got=api().reverse_step(x,jnp.zeros_like(x),y,m,jnp.float32(.9999),jnp.float32(1.),jnp.ones_like(x)*99,jnp.ones_like(x)*99)
    np.testing.assert_allclose(got,np.array([[[[.3],[-1.]]]]),atol=1e-7)


def test_tiny_sampler_exact_prng_stream_time_index_and_trace():
    mod=api();shape=(2,2,2,1)
    y=np.zeros(shape,np.float32);y[:,0,0,0]=.4
    mask=np.zeros(shape,bool);mask[:,0,0,0]=True
    # This predictor makes a t versus t-1 indexing defect observable.
    def predictor(x,t):return jnp.broadcast_to(t[:,None,None,None]/1000.,x.shape)
    samples,trace=mod.repaint_sample(predictor,y,mask,steps=3,repeats=2,seed=17)
    ts=np.rint(np.linspace(1,1000,3)).astype(int)[::-1]
    pairs=[(int(t),int(s),c,c<1 and s!=0) for t,s in zip(ts,np.r_[ts[1:],0]) for c in range(1 if s==0 else 2)]
    assert len(trace['t'])==5
    key,ki=jax.random.split(jax.random.PRNGKey(17));x=jax.random.normal(ki,shape)
    # Use independent float64 schedule, with tolerances for float32 scan arithmetic.
    abar=np.r_[1.,np.cumprod(1.-np.linspace(.0001,.02,1000,dtype=np.float64))]
    for index,(t,s,c,again) in enumerate(pairs):
        np.testing.assert_allclose(trace['x_t'][index],np.asarray(x)[0,:,:,0],atol=2e-5)
        key,kz,ke,kf=jax.random.split(key,4)
        z=np.asarray(jax.random.normal(kz,shape));eta=np.asarray(jax.random.normal(ke,shape));f=np.asarray(jax.random.normal(kf,shape))
        eps=np.full(shape,(t-1)/1000.,np.float32)
        np.testing.assert_allclose(trace['epsilon'][index],eps[0,:,:,0],atol=1e-7)
        bt=abar[t];bs=abar[s];a=bt/bs
        clean=np.clip((np.asarray(x)-np.sqrt(1-bt)*eps)/np.sqrt(bt),-1,1)
        mean=np.sqrt(bs)*(1-a)/(1-bt)*clean+np.sqrt(a)*(1-bs)/(1-bt)*np.asarray(x)
        result=np.where(mask,np.sqrt(bs)*y+np.sqrt(1-bs)*eta,mean+np.sqrt((1-bs)*(1-a)/(1-bt))*z)
        np.testing.assert_allclose(trace['x_s'][index],result[0,:,:,0],atol=3e-5)
        x=jnp.asarray(np.sqrt(a)*result+np.sqrt(1-a)*f if again else result,dtype=jnp.float32)
    np.testing.assert_allclose(samples,np.asarray(x),atol=3e-5)
    np.testing.assert_array_equal(np.asarray(samples)[mask],y[mask])
    assert np.asarray(samples).dtype==np.float32
