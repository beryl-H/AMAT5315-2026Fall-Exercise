from pathlib import Path
import sys,os,json,hashlib,subprocess,time,datetime,resource,shutil
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);args=ap.parse_args()
r=Path(__file__).resolve().parents[1];s=r/'artifacts'/'controller'/('memory-'+args.attempt);s.mkdir(parents=True,exist_ok=False)
# No other part's in-progress tests are included.
with (s/"memory-release-tests.log").open("x") as f:p=subprocess.run([str(r/".venv/bin/python"),"-m","pytest","tests/test_memory_sweep.py","tests/test_memory_stores.py","-q"],cwd=r,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,(s/"memory-release-tests.log").read_text()
a=r/'artifacts'/'part3'/args.attempt;a.mkdir(parents=True,exist_ok=False)
code=r/"src/memory_sweep.py";h=hashlib.sha256(code.read_bytes()).hexdigest();shutil.copy2(code,a/"memory_sweep-used-source.py")
settings={"attempt":args.attempt,"d_values":[16,32,64],"M_values":[2**p for p in range(3,17)],"seeds":[101,102,103],"stores":["outer","ls","relu1","relu2","relu3"],"query_tile":256,"queries":"every stored key of every store; all M, no subsets","aggregate":"median over queried keys per draw, then median3 draws","rng":"full65536 K then full65536 V independently drawn sequentially per d and seed; smaller M prefixes","blas_requested_threads":os.environ.get("OPENBLAS_NUM_THREADS"),"cpu_affinity":sorted(os.sched_getaffinity(0)),"source_sha256":h,"source_snapshot":"memory_sweep-used-source.py","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"actor":"Pi authored exact sweep and tests; Codex sole formal dispatch","expected_progress_records":630,"estimated_runtime_seconds":871,"neural_training_runs":0}
(a/"settings.json").write_text(json.dumps(settings,indent=2)+"\n");(s/"memory-dispatch.json").write_text(json.dumps(settings|{"pid":os.getpid()},indent=2)+"\n")
sys.path.insert(0,str(r));from src.memory_sweep import run_sweep
start=time.time();result=run_sweep(settings["d_values"],settings["M_values"],settings["seeds"],str(a),formal=True)
records=[json.loads(x) for x in (a/"progress.jsonl").read_text().splitlines()];assert len(records)==630
assert len({(x["case"]["d"],x["case"]["M"],x["case"]["seed"],x["case"]["store"]) for x in records})==630
assert hashlib.sha256(code.read_bytes()).hexdigest()==h,"live source changed; used source preserved"
out={"status":"COMPLETE","elapsed_seconds":time.time()-start,"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"source_sha256":h,"progress_records":len(records),"source_unchanged":True,"capacity":result["capacity"]};(a/"completion-receipt.json").write_text(json.dumps(out,indent=2)+"\n");(s/"memory-execution-receipt.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out),flush=True)
