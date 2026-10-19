"""Frozen-prior RePaint-style inference; image inputs are observations and mask."""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
from inpainting import validate_observations


def transition_grid(steps,repeats):
    if not 2<=steps<=1000 or repeats<1:
        raise ValueError('steps must be 2..1000 and repeats positive')
    levels=np.rint(np.linspace(1,1000,steps)).astype(int)[::-1]
    rows=[]
    for t,s in zip(levels,np.r_[levels[1:],0]):
        count=1 if s==0 else repeats
        rows.extend((t,s,c,c<count-1) for c in range(count))
    return tuple(np.array([r[i] for r in rows],dtype=bool if i==3 else np.int32) for i in range(4))


def schedule():
    beta=jnp.linspace(.0001,.02,1000,dtype=jnp.float32)
    return jnp.concatenate([jnp.ones(1,jnp.float32),jnp.cumprod(jnp.float32(1)-beta)])


def reverse_step(x,epsilon,observations,mask,abar_t,abar_s,z,eta):
    a=abar_t/abar_s
    clean=jnp.clip((x-jnp.sqrt(1-abar_t)*epsilon)/jnp.sqrt(abar_t),-1,1)
    mean=jnp.sqrt(abar_s)*(1-a)/(1-abar_t)*clean+jnp.sqrt(a)*(1-abar_s)/(1-abar_t)*x
    variance=(1-abar_s)*(1-a)/(1-abar_t)
    proposal=mean+jnp.sqrt(variance)*z
    known=jnp.sqrt(abar_s)*observations+jnp.sqrt(1-abar_s)*eta
    return jnp.where(mask,known,proposal)


@lru_cache(maxsize=4)
def _compiled_sampler(predict,shape,steps,repeats):
    t_values,s_values,cycles,again=transition_grid(steps,repeats)
    abar=schedule()
    @jax.jit
    def run(key,y,mask):
        key,ki=jax.random.split(key)
        initial=jax.random.normal(ki,shape,dtype=jnp.float32)
        def body(carry,row):
            key,x=carry;t,s,repeat=row
            key,kz,ke,kf=jax.random.split(key,4)
            z=jax.random.normal(kz,shape,dtype=jnp.float32)
            eta=jax.random.normal(ke,shape,dtype=jnp.float32)
            forward=jax.random.normal(kf,shape,dtype=jnp.float32)
            eps=predict(x,jnp.full((shape[0],),t-1,dtype=jnp.int32))
            xs=reverse_step(x,eps,y,mask,abar[t],abar[s],z,eta)
            a=abar[t]/abar[s]
            next_state=jax.lax.cond(repeat,lambda _:jnp.sqrt(a)*xs+jnp.sqrt(1-a)*forward,lambda _:xs,operand=None)
            record=(x[0,:,:,0],eps[0,:,:,0],xs[0,:,:,0])
            return (key,next_state),record
        (_,final),record=jax.lax.scan(body,(key,initial),(jnp.array(t_values),jnp.array(s_values),jnp.array(again)))
        return final,record
    return run


def repaint_sample(predict,observations,mask,steps,repeats,seed):
    y,m=validate_observations(observations,mask)
    if y.ndim!=4 or y.shape[-1]!=1:raise ValueError('sampler expects [batch,height,width,1]')
    if y.shape[0]<1:raise ValueError('sampler batch must not be empty')
    y=jnp.asarray(y,dtype=jnp.float32);m=jnp.asarray(m,dtype=bool)
    run=_compiled_sampler(predict,tuple(y.shape),steps,repeats)
    result,record=run(jax.random.PRNGKey(seed),y,m)
    result=np.asarray(result)
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite sampler result')
    t,s,cycle,_=transition_grid(steps,repeats)
    trace=dict(t=t,s=s,cycle=cycle,x_t=np.asarray(record[0]),epsilon=np.asarray(record[1]),x_s=np.asarray(record[2]))
    return result,trace


def load_prior(checkpoint):
    """Reuse Week 6's loader and forward pass verbatim, with frozen EMA leaves."""
    source=Path(__file__).resolve().parent.parent/'week6/src'
    if not source.is_dir():raise FileNotFoundError('Week 6 model source is required')
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    from diffusion.checkpoint import load_checkpoint
    from diffusion.model import unet_apply
    config,params,settings=load_checkpoint(checkpoint)
    if config.image_size!=32 or config.in_ch!=1 or settings.get('case')!='mnist':
        raise ValueError('expected the Week 6 MNIST prior')
    if settings.get('t_steps')!=1000 or settings.get('beta')!=[.0001,.02]:
        raise ValueError('checkpoint noise schedule differs from the exercise')
    return lambda x,t:unet_apply(params,config,x,t)
