from pathlib import Path
import sys,json,numpy as np
r=Path("/home/keria_h/AMAT5315-2026Fall-Exercise/week8");sys.path.insert(0,str(r))
import jax
jax.config.update("jax_enable_x64",False)
import jax.numpy as jnp
import grok
p=grok.load_params(r/"artifacts/part4/formal-decoder-02/main-s0/checkpoint-final.npz")
d=grok.build_dataset();sp=grok.fixed_split(.3);x=jnp.asarray(d["inputs"][sp["train"]]);y=jnp.asarray(d["targets"][sp["train"]]);N=len(y)
f=lambda p:grok.forward(p,x)[:,2,:]
z,pb=jax.vjp(f,p)
# Independent analytic CE derivative: target = minus sum of other probabilities.
# Unlike p_true-1, this does not subtract nearly equal numbers.
q=jnp.exp(jax.nn.log_softmax(z));q=q.at[jnp.arange(N),y].set(0);q=q.at[jnp.arange(N),y].set(-q.sum(axis=1));g_analytic=pb(q/N)[0]
def equivalent(p):
 z=f(p);v=jnp.where(jnp.arange(97)[None,:]==y[:,None],-jnp.inf,z)
 return jnp.logaddexp(jnp.asarray(0,dtype=z.dtype),jax.nn.logsumexp(v,axis=1)-z[jnp.arange(N),y]).mean()
g_stable=jax.grad(equivalent)(p);g_orig=jax.grad(lambda a:grok.loss_acc(a,x,y)[0])(p)
def flatten(t):return np.concatenate([np.asarray(v,dtype=np.float64).ravel() for v in jax.tree_util.tree_leaves(t)])
a,b,c=map(flatten,(g_analytic,g_stable,g_orig))
jax.config.update("jax_enable_x64",True);p64=jax.tree_util.tree_map(lambda v:v.astype(jnp.float64),p);ref=flatten(jax.grad(lambda a:grok.loss_acc(a,x,y)[0])(p64))
err=lambda v:float(np.linalg.norm(v-ref)/np.linalg.norm(ref))
out={"status":"PASS","scope":"independent local numerical-mechanism verification only, not training causation","analytic_vs_stable_relL2":float(np.linalg.norm(a-b)/np.linalg.norm(b)),"original_vs_float64_relL2":err(c),"stable_vs_float64_relL2":err(b),"analytic_target_complement_vs_float64_relL2":err(a),"optimizer_updates":0}
assert out["analytic_vs_stable_relL2"]<1e-3
assert err(a)<1e-3 and err(b)<1e-3
s=Path("/mnt/d/Keria/Course/AMAT5315/week8-support/part4-seed0-investigation/minimal-diagnostics-03");(s/"independent-cancellation-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
follow=r/"diagnostics/minimal-diagnostics-03/followup-ce-cancellation/results/followup_ce_cancellation.json";v=json.loads(follow.read_text());print(json.dumps({"followup_verdict":v["verdict_engineering"],"harness":v["harness_gate"],"final_checks":v["final_checks"],"snapshots":{k:{"original":z["grad_f32_orig_vs_f64_orig"]["relL2"],"stable":z["grad_f32_stable_vs_f64_orig"]["relL2"],"improvement":z["improvement_ratio"],"criterion_met":z["criterion_met"]} for k,z in v["snapshots"].items()}}))
