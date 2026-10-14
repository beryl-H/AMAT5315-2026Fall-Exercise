"""Independent full-array acceptance: NumPy reference, actual library attention, matched queries and classical stability."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,gzip,struct,time
import numpy as np
import jax,jax.numpy as jnp
from ebm.modern_hopfield import modern_update,attention_update
root=Path(__file__).resolve().parent.parent; support=root/'evidence'
metrics=json.loads((root/'evidence/part4_metrics.json').read_text());assert len(metrics['points'])==6
runs=[p for p in sorted((root/'artifacts/part4').glob('evaluation-*')) if len(list(p.glob('point_P*.npz')))==6];assert runs
run=runs[-1]
buf=gzip.open(root/'inputs/mnist/train-images-idx3-ubyte.gz','rb').read();assert struct.unpack('>4I',buf[:16])==(2051,60000,28,28)
images=np.frombuffer(buf,np.uint8,offset=16).reshape(60000,784)
betas=[.001,.003,.01,.03,.1,.3,1.];report=[];global_diff=0.;wrong_diff=0.
for point,P in zip(metrics['points'],[3,10,30,100,300,1000]):
 assert point['P']==P
 d=np.load(run/f'point_P{P}.npz');X=(2*(images[:P]>=128).astype(np.int32)-1).T;Q=d['queries'];targets=np.arange(100)%P
 np.testing.assert_array_equal(d['targets'],targets);assert Q.shape==(100,784) and np.isin(Q,[-1,1]).all()
 np.testing.assert_array_equal(Q[:,:392],X[:,targets].T[:,:392])
 key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026),P),1)
 random_spins=np.asarray(jax.vmap(lambda k:jax.random.bernoulli(k,.5,(784,)))(jax.random.split(key,100)),np.int32)*2-1
 np.testing.assert_array_equal(Q[:,392:],random_spins[:,392:])
 diagnostics={'targets':targets};rates={};maxref=0.;perbeta={}
 def criterion(out):
  sg=np.sign(out).astype(np.int32);ovs=sg@X;own=ovs[np.arange(100),targets];nearest=ovs.argmax(1);success=(100*own>=95*784)&(nearest==targets)
  return own/784,nearest,success
 for b in betas:
  out=d[f'modern_out_beta{b}'];assert out.dtype==np.float32 and out.shape==(100,784)
  logits=b*(Q.astype(float)@X.astype(float));weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True);ref=weights@X.T
  maxref=max(maxref,float(abs(out-ref).max()));np.testing.assert_allclose(out,ref,atol=8e-6,rtol=2e-5)
  actual=np.asarray(jax.vmap(lambda q:modern_update(X,q,b))(jnp.asarray(Q)))
  att=np.asarray(jax.vmap(lambda q:attention_update(X,q,b))(jnp.asarray(Q)))
  np.testing.assert_array_equal(actual,out)
  delta=float(abs(actual-att).max());assert delta<1e-5;perbeta[str(b)]=delta;global_diff=max(global_diff,delta)
  ov,near,success=criterion(out);rates[str(b)]=float(success.mean());np.testing.assert_allclose(rates[str(b)],point['modern_success_frac'][str(b)],atol=0)
  diagnostics.update({f'beta{b}_overlap':ov,f'beta{b}_nearest':near,f'beta{b}_success':success})
  if P==100 and b==.3:
   default=np.asarray(jax.nn.dot_product_attention(jnp.asarray(Q)[:,None,:],jnp.asarray(X.T,dtype=jnp.float32)[:,None,:],jnp.asarray(X.T,dtype=jnp.float32)[:,None,:]))[:,0,:]
   wrong_diff=float(abs(default-out).max());assert wrong_diff>.1
 assert rates['0.3']>=.95
 classic=d['classical_out'].astype(np.int32);assert d['classical_converged'].all()
 K=X@X.T;np.fill_diagonal(K,0);assert (classic*(classic@K)>=0).all()
 ov,near,success=criterion(classic);assert success.sum()==0 and point['classical_success_frac']==0
 diagnostics.update(classical_overlap=ov,classical_nearest=near,classical_success=success)
 np.savez_compressed(run/f'independent_diagnostics_P{P}.npz',**diagnostics)
 report.append({'P':P,'modern_beta0.3_success':rates['0.3'],'classical_success':0.,'max_numpy_reference_difference':maxref,'attention_diff_by_beta':perbeta,'all_classical_states_stable':True})
 np.testing.assert_allclose(max(perbeta.values()),point['max_attention_diff'],atol=0)
np.testing.assert_allclose(global_diff,metrics['global_max_attention_diff'],atol=0)
# Recover complete immutable experiment record beside its raw arrays.
(run/'independent_verified_summary.json').write_text(json.dumps(metrics,indent=2))
r={'artifact_directory':str(run.relative_to(root)),'queries_verified':600,'modern_outputs_verified':4200,'global_max_attention_difference':global_diff,'default_scale_negative_control_max_difference':wrong_diff,'points':report}
(support/'part4_independent_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
