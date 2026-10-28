from pathlib import Path
import json,hashlib,numpy as np
root=Path("/home/keria_h/AMAT5315-2026Fall-Exercise/week8")
a=root/"artifacts/part4/formal-decoder-02/main-s0";b=root/"artifacts/part4/seed0-replay-investigation-01/main-s0"
old=[json.loads(x) for x in (a/"train-log.jsonl").read_text().splitlines()];new=[json.loads(x) for x in (b/"train-log.jsonl").read_text().splitlines()]
diffs=[]
for o,n in zip(old,new):
 for k in o:
  if k!="unix_ts" and o[k]!=n[k]:diffs.append({"step":n["step"],"key":k,"original":o[k],"replay":n[k]})
weights={}
for n in ("init","memorization","final"):
 p=b/f"checkpoint-{n}.npz"
 if p.exists():
  x=np.load(a/p.name);y=np.load(p);weights[n]=all(np.array_equal(x[k],y[k]) for k in x.files) and set(x.files)==set(y.files)
manifest=json.loads(Path("/mnt/d/Keria/Course/AMAT5315/week8-support/part4-seed0-investigation/original-artifact-hashes.json").read_text())
changed=[p for p,h in manifest["files"].items() if hashlib.sha256((root/p.replace(chr(92),'/')).read_bytes()).hexdigest()!=h]
result={"status":"PASS" if not diffs and all(weights.values()) and not changed else "FAIL","complete":(b/"metrics.json").exists(),"compared_log_points":len(new),"last_step":new[-1]["step"],"numeric_differences":diffs,"checkpoint_arrays_bitwise_equal":weights,"original_artifacts_checked":len(manifest["files"]),"original_artifacts_changed":changed,"last_log":new[-1]}
Path("/mnt/d/Keria/Course/AMAT5315/week8-support/part4-seed0-investigation/independent-replay-comparison.json").write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result["status"]=="PASS"
