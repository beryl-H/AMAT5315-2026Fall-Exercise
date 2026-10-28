from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,"/home/keria_h/AMAT5315-2026Fall-Exercise/week8")
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import grok
rng=np.random.default_rng(20261028);cases=[]
for wd in (0.,1.):
 p=rng.normal(size=(16,16));m=np.zeros_like(p);v=m.copy();jp={"x":jnp.asarray(p)};st=grok.grok_adamw_init(jp);errors={"params":0.,"m":0.,"v":0.}
 for step in range(1,201):
  g=rng.normal(size=p.shape);m=.9*m+.1*g;v=.98*v+.02*g*g
  p=(1-.003*wd)*p-.003*(m/(1-.9**step))/(np.sqrt(v/(1-.98**step))+1e-8)
  jp,st=grok.grok_adamw_step(jp,{"x":jnp.asarray(g)},st,lr=.003,step=step,weight_decay=wd)
  for k,a,b in (("params",jp["x"],p),("m",st["m"]["x"],m),("v",st["v"]["x"],v)):
   errors[k]=max(errors[k],float(np.max(np.abs(np.asarray(a)-b))))
 cases.append({"wd":wd,"steps":200,"max_absolute_errors":errors})
assert max(v for c in cases for v in c["max_absolute_errors"].values())<1e-12
out={"status":"PASS","dtype":"float64 on both implementations","tolerance":1e-12,"compared":"parameters and both moment states at every step","cases":cases}
Path("/mnt/d/Keria/Course/AMAT5315/week8-support/part4-seed0-investigation/independent-optimizer.json").write_text(json.dumps(out,indent=2));print(json.dumps(out))
