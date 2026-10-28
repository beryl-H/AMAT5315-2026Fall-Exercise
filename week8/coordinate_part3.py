# Copied verbatim from week8-support/coordinate_part3.py (2026-10-28) into
# the repository as coordination-only source, by pi (volcengine-plan /
# glm-5-3-flash) at the direction of pi-task-08 TASK B. AUTHORSHIP: Codex
# coordination layer (dispatch logic, receipts, prediction seal checks);
# the model workers it dispatches were authored through native WSL Pi.
# The ONLY functional edit is the repository-root line below: the hardcoded
# absolute path is replaced by this file's parent directory so the wrapper
# is portable; no frozen model/fit/scorer logic is altered.
"""Codex coordination only: dispatch Pi-authored workers, preserve every run."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1])/1024
    except FileNotFoundError: pass
    return 0.
a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--widths',required=True);a.add_argument('--steps',type=int,default=1500);a.add_argument('--workers',type=int,default=4);a.add_argument('--prediction');args=a.parse_args()
assert 1<=args.workers<=4
root=Path(__file__).resolve().parent;os.chdir(root)
widths=list(map(int,args.widths.split(',')));assert all(w in [8,12,16,24,32,48,64,96,128,192] for w in widths)
formal=args.steps==1500;assert formal or args.steps in [1,2,3]
files=['transformer.py','run_part2.py','scaling.py','sweep.py'];frozen={n:sha(root/n) for n in files}
receipt=None
if 192 in widths:
    assert formal and args.prediction
    receipt=json.loads(Path(args.prediction).read_text());payload={k:v for k,v in receipt.items() if k!='sha256'}
    assert sha(Path(args.prediction))
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==receipt['sha256']
    assert receipt['target_width']==192 and receipt['target_N']==890112
    assert len(receipt['protocol']['fit_run_manifest_shas'])==18
    for n,h in frozen.items(): assert receipt['protocol']['frozen_hashes'][n]==h,(n,'source changed after prediction')
    timestamp=datetime.datetime.strptime(receipt['utc_timestamp'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    assert timestamp<time.time()
out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'SPEC-frozen.md').write_bytes((root/'SPEC.md').read_bytes())
state={'actor':'Codex dispatch manager; model workers authored through native WSL Pi','formal':formal,'started_utc':utc(),'started_unix':time.time(),'steps':args.steps,'widths':widths,'seeds':[1,2],'source_hashes':frozen,'spec_sha256':sha(out/'SPEC-frozen.md'),'max_worker_rss_mb':0.,'completed':[],'active':[],'status':'RUNNING'}
if receipt: state['prediction_sha256']=receipt['sha256']
(out/'coordinator-source.py').write_bytes(Path(__file__).read_bytes())
queue=[(w,s) for w in widths for s in [1,2]];active={};failed=False
while queue or active:
    for core in range(args.workers):
        if core in active or not queue or failed: continue
        for n,h in frozen.items(): assert sha(root/n)==h,(n,'source changed during sweep')
        w,s=queue.pop(0);d=out/f'w{w:03d}-s{s}'
        cmd=['env','JAX_PLATFORMS=cpu','taskset','-c',str(core),'.venv/bin/python','sweep.py','--width',str(w),'--seed',str(s),'--steps',str(args.steps),'--out',str(d)]
        stdout=open(out/f'w{w:03d}-s{s}.stdout','x');stderr=open(out/f'w{w:03d}-s{s}.stderr','x')
        p=subprocess.Popen(cmd,cwd=root,stdout=stdout,stderr=stderr)
        active[core]={'p':p,'w':w,'s':s,'d':d,'cmd':cmd,'stdout':stdout,'stderr':stderr,'started':time.time()}
        print('START',w,s,'core',core,'pid',p.pid,flush=True)
    total=sum(rss(x['p'].pid) for x in active.values());state['max_worker_rss_mb']=max(total,state['max_worker_rss_mb'])
    if total>5800:
        failed=True;state['memory_limit_failure']=total
        for x in active.values(): x['p'].terminate()
    for core,x in list(active.items()):
        rc=x['p'].poll()
        if rc is None: continue
        x['stdout'].close();x['stderr'].close()
        rec={'width':x['w'],'seed':x['s'],'core':core,'pid':x['p'].pid,'command':x['cmd'],'started_unix':x['started'],'finished_unix':time.time(),'exit_code':rc}
        if rc==0:
            try:
                m=json.loads((x['d']/'metrics.json').read_text());s=json.loads((x['d']/'settings.json').read_text())
                assert m['steps']==args.steps and m['n_scored']==111539 and s['heads']==2
                assert s['source_hashes']==frozen and m['checkpoint_sha256']==sha(x['d']/'weights.npz')
                assert s['formal']==formal
                if receipt: assert timestamp<s['started_unix_ts']
                rec.update(loss=m['full_heldout_loss'],metrics_sha256=sha(x['d']/'metrics.json'),settings_sha256=sha(x['d']/'settings.json'))
            except Exception as exc:
                failed=True;rec['verification_failure']=str(exc)
        else: failed=True
        state['completed'].append(rec);del active[core];print('FINISH',rec,flush=True)
    state['active']=[{'width':x['w'],'seed':x['s'],'pid':x['p'].pid,'core':c,'rss_mb':rss(x['p'].pid)} for c,x in active.items()]
    state['queued']=len(queue);state['updated_utc']=utc();(out/'manager-status.json').write_text(json.dumps(state,indent=2)+'\n')
    if failed and not active: break
    if active: time.sleep(1)
state['status']='FAILED' if failed else 'COMPLETE';state['finished_utc']=utc();state['finished_unix']=time.time();(out/'manager-status.json').write_text(json.dumps(state,indent=2)+'\n')
print(state['status'],'completed',len(state['completed']),'max RSS MB',state['max_worker_rss_mb'],flush=True)
raise SystemExit(1 if failed else 0)
