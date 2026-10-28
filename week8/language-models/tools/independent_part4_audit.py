from pathlib import Path
import json,numpy as np,hashlib,shutil
from scipy.special import softmax
r=Path(__file__).resolve().parents[1];s=r/"evidence/audits";d=json.loads((r/"evidence/part4-results.json").read_text());g=json.loads((r/"evidence/glyph.json").read_text());v=np.array(g["mask"],float).ravel();assert v.size==1024 and set(np.unique(v))=={-1.,1.} and np.linalg.norm(v)==32;assert g["character"]=="算"
raw={};worst=0
for seed in range(531504,531524):
 for M in (64,256,1024,4096):
  rng=np.random.default_rng(seed);K=rng.normal(size=(M,64));K/=np.linalg.norm(K,axis=1,keepdims=True);V=(2*rng.integers(0,2,size=(M,1024))-1).astype(float);V[0]=v;scores=K@K[0]
  weights={"outer":scores,"relu":np.maximum(scores,0),"relu2":np.maximum(scores,0)**2,"softmax":softmax(8*scores)}
  for name,w in weights.items():
   other=w@V-w[0]*v;R=float(np.sqrt(other@other)/(abs(w[0])*32));key=f"{name}|M={M}|seed={seed}";raw[key]=R;worst=max(worst,abs(R-d["all_raw_ISR"][key]));assert np.isclose(R,d["all_raw_ISR"][key],rtol=1e-11,atol=1e-12),key
for key,x in d["medians_20draws"].items():
 st,m=key.split("|M=");computed=np.median([raw[f"{st}|M={m}|seed={seed}"] for seed in range(531504,531524)]);assert abs(computed-x)<1e-12
out={"status":"PASS","independently_recomputed_cells":len(raw),"worst_absolute_ISR_difference":worst,"glyph":"算","glyph_norm":32,"mask_sha256_float64":hashlib.sha256(v.tobytes()).hexdigest(),"mandatory_gates":d["gates"],"softmax_max_all_80_cells":max(x for k,x in raw.items() if k.startswith("softmax|")),"medians":d["medians_20draws"],"training_runs":0};(s/"independent-part4-audit.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n");print(json.dumps(out,ensure_ascii=True))
for a,b in (("part4-recall.png","part4-recall-original.png"),("glyph-preview.png","glyph-preview-original.png")):
 p=s/b
 if not p.exists():shutil.copy2(r/"evidence"/a,p)
