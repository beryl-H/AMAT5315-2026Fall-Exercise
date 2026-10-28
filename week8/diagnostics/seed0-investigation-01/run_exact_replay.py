from pathlib import Path
import hashlib,sys,json
root=Path("/home/keria_h/AMAT5315-2026Fall-Exercise/week8")
assert hashlib.sha256((root/"grok.py").read_bytes()).hexdigest()=="d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447"
sys.path.insert(0,str(root))
import grok
m=grok.run_one(seed=0,fraction=.30,wd=1.,max_steps=10000,log_every=100,outdir=root/"artifacts/part4/seed0-replay-investigation-01/main-s0")
print(json.dumps({k:m[k] for k in ("seed","steps_run","memorization_step","grok_step")}))
