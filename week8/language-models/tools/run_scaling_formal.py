from pathlib import Path
import sys,os,json,hashlib,time,datetime,subprocess,shutil,itertools
import numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);args=ap.parse_args()
r=Path(__file__).resolve().parents[1];s=r/'artifacts'/'controller'/('scaling-'+args.attempt);s.mkdir(parents=True,exist_ok=False)
with (s/'scaling-release-tests.log').open('x') as f:
 p=subprocess.run([str(r/'.venv/bin/python'),'-m','pytest','tests/test_scaling.py','-q'],cwd=r,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,(s/'scaling-release-tests.log').read_text()
sys.path.insert(0,str(r));from src import scaling as m
source=r/'src/scaling.py';h=hashlib.sha256(source.read_bytes()).hexdigest()
assert h=='1a33e2c1e3a13805654e0d54d152b082560726bd1ffdaf3bbbd2475dd948df3f'
a=r/'artifacts'/'part6'/args.attempt;a.mkdir(parents=True,exist_ok=False);shutil.copy2(source,a/'scaling-used-source.py');shutil.copy2(Path(__file__),a/'dispatch-used-source.py')
N,C,L=m.load_runs(r/'data/chinchilla-runs.csv');N,D,L=m.filter_runs(N,C,L);logL=np.log(L);assert len(L)==240
settings={'actor':'Pi-authored frozen analytic fitter; Codex sole formal orchestration','attempt':args.attempt,'source_sha256':h,'n_original':245,'n_filtered':240,'filter':'drop5 highest loss once','objective':'mean Huber(logLpred-logLobs, delta=.001)/delta^2','bounds':m.BOUNDS,'optimizer_options':m.FIT_OPTIONS,'grid':m.GRID,'multistarts':243,'bootstrap_seeds':list(range(400)),'bootstrap':'single fit per same resample from best full fit; no automatic retries or resampling','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'data_sha256':hashlib.sha256((r/'data/chinchilla-runs.csv').read_bytes()).hexdigest()}
(a/'settings.json').write_text(json.dumps(settings,indent=2)+'\n');(s/'scaling-dispatch.json').write_text(json.dumps(settings|{'pid':os.getpid()},indent=2)+'\n')
t0=time.time()
def record(res,start,**extra):
 g=np.array(res.jac);pg=g.copy()
 for j,(lo,hi) in enumerate(m.BOUNCE_LIST):
  if (res.x[j]<=lo+1e-12 and g[j]>0) or (res.x[j]>=hi-1e-12 and g[j]<0):pg[j]=0
 return dict(extra,start_theta=np.array(start).tolist(),end_theta=res.x.tolist(),params=m.theta_to_params(res.x),fun=float(res.fun),success=bool(res.success),status=int(res.status),message=str(res.message),nit=int(res.nit),nfev=int(res.nfev),gradient=g.tolist(),raw_gradient_L2=float(np.linalg.norm(g)),projected_gradient_infinity=float(abs(pg).max()))
def append(name,x):
 with (a/name).open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
fits=[]
for i,th in enumerate(m.grid_starts()):
 res=m.fit_once(N,D,logL,th);rec=record(res,th,start_index=i);append('multistart.jsonl',rec);fits.append((res,rec))
 if (i+1)%25==0:print('multistart',i+1,'/243',flush=True)
valid=[x for x in fits if x[0].success and np.isfinite(x[0].fun)];assert valid,'No converged fit'
best,brecord=min(valid,key=lambda x:x[0].fun);overall=min(fits,key=lambda x:x[0].fun)
assert best.fun<=overall[0].fun+1e-8,'Better unconverged solution needs review; all traces retained'
p=m.theta_to_params(best.x);printed=np.array([np.log(1.69),np.log(406.4),np.log(410.7),.34,.28]);Ns,Ds=m.derive_optimal(p['alpha'],p['A'],p['beta'],p['B'],5.76e23)
base={'best':brecord,'successful_starts':len(valid),'best_overall_status':overall[1],'mean_log_residual_own':m.mean_log_residual(best.x,N,D,logL),'mean_log_residual_printed':m.mean_log_residual(printed,N,D,logL),'optimal_at_C5.76e23':{'N':float(Ns),'D':float(Ds),'D_over_N':float(Ds/Ns)},'elapsed_fit_seconds':time.time()-t0}
(a/'full-fit.json').write_text(json.dumps(base,indent=2)+'\n');np.savez(a/'filtered-data-and-fit.npz',N=N,D=D,L=L,theta=best.x)
boots=[]
for seed in range(400):
 idx=np.random.default_rng(seed).integers(0,240,size=240);res=m.fit_once(N[idx],D[idx],logL[idx],best.x)
 rec=record(res,best.x,seed=seed,resample_indices=idx.tolist());append('bootstrap.jsonl',rec);boots.append(rec)
 if (seed+1)%50==0:print('bootstrap',seed+1,'/400',flush=True)
failed=[x['seed'] for x in boots if not x['success']]
params=np.array([[b['params'][k] for k in ['E','A','B','alpha','beta']] for b in boots]);ratios=[]
for b in boots:
 q=b['params'];nn,dd=m.derive_optimal(q['alpha'],q['A'],q['beta'],q['B'],5.76e23);ratios.append(float(dd/nn))
gates={'alpha_032_038':.32<=p['alpha']<=.38,'beta_032_042':.32<=p['beta']<=.42,'E_175_190':1.75<=p['E']<=1.90,'own_mean_residual_abs_le_0005':abs(base['mean_log_residual_own'])<=.005,'printed_mean_gt_001':base['mean_log_residual_printed']>.01,'optimal_ratio_10_30':10<=Ds/Ns<=30,'all_bootstrap_optimizer_success':not failed,'compute_identity':bool(np.isclose(6*Ns*Ds,5.76e23,rtol=1e-12))}
out=base|{'parameter_std_ddof1':dict(zip(['E','A','B','alpha','beta'],params.std(axis=0,ddof=1).tolist())),'optimal_ratio_bootstrap_95pct':np.percentile(ratios,[2.5,97.5]).tolist(),'bootstrap_failures':failed,'bootstrap_count':400,'mandatory_gates':gates,'elapsed_seconds':time.time()-t0,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==h,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(a/'results.json').write_text(json.dumps(out,indent=2)+'\n');(s/'scaling-execution-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
