# Authored by Codex for coordination of Pi-authored model workers.
# Portable repository copy; existing live run retains its original source receipt.
"""Codex coordination layer for Pi-authored grokking workers; no model code."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])/1024
    except FileNotFoundError:pass
    return 0.
a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--workers',type=int,default=4);args=a.parse_args();assert 1<=args.workers<=4
root=Path(__file__).resolve().parent;os.chdir(root)
frozen={n:sha(root/n) for n in ['grok.py','SPEC.md','run_part4.py','PART4_DECODER_PROTOCOL.md'] if (root/n).exists()}
jobs=[{'name':f'main-s{s}','seed':s,'fraction':.3,'wd':1.,'max_steps':10000,'permuted_labels':False} for s in range(5)]
jobs += [{'name':f'wd0-s{s}','seed':s,'fraction':.3,'wd':0.,'max_steps':10000,'permuted_labels':False} for s in range(3)]
jobs += [{'name':f'frac50-s{s}','seed':s,'fraction':.5,'wd':1.,'max_steps':10000,'permuted_labels':False} for s in range(3)]
jobs += [{'name':'random-labels-s0','seed':0,'fraction':.3,'wd':1.,'max_steps':3000,'permuted_labels':True}]
if args.smoke:
    jobs=[jobs[0],jobs[5]]
    for j in jobs:j['max_steps']=3
out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'SPEC-frozen.md').write_bytes((root/'SPEC.md').read_bytes());(out/'coordinator-source.py').write_bytes(Path(__file__).read_bytes())
state={'actor':'Codex dispatch; grokking worker and model authored via native WSL Pi plus explicitly recorded Codex wd threading repair','formal':not args.smoke,'started_unix':time.time(),'started_utc':utc(),'source_hashes':frozen,'jobs':jobs.copy(),'status':'RUNNING','completed':[],'active':[],'max_combined_worker_rss_mb':0.}
active={};failed=False
while jobs or active:
    for slot in range(args.workers):
        if slot in active or not jobs or failed:continue
        for n,h in frozen.items():assert sha(root/n)==h,(n,'source drift')
        job=jobs.pop(0);d=out/job['name'];kwargs={k:v for k,v in job.items() if k!='name'};kwargs['log_every']=1 if args.smoke else 100
        code='from pathlib import Path; import grok,json; m=grok.run_one(outdir=Path('+repr(str(d))+'), **'+repr(kwargs)+'); print(json.dumps({k:m[k] for k in ["seed","steps_run","grok_step","memorization_step"]}))'
        core=slot+4;cmd=['env','JAX_PLATFORMS=cpu','taskset','-c',str(core),'.venv/bin/python','-c',code]
        so=open(out/(job['name']+'.stdout'),'x');se=open(out/(job['name']+'.stderr'),'x');p=subprocess.Popen(cmd,cwd=root,stdout=so,stderr=se)
        active[slot]={'p':p,'job':job,'d':d,'cmd':cmd,'so':so,'se':se,'start':time.time()};print('START',job['name'],'core',core,'pid',p.pid,flush=True)
    total=sum(rss(x['p'].pid) for x in active.values())
    state['max_combined_worker_rss_mb']=max(total,state['max_combined_worker_rss_mb'])
    if total>5800:
        failed=True;state['memory_cap_failure_mb']=total
        for x in active.values():x['p'].terminate()
    for slot,x in list(active.items()):
        rc=x['p'].poll()
        if rc is None:continue
        x['so'].close();x['se'].close();job=x['job'];d=x['d'];rec={'name':job['name'],'pid':x['p'].pid,'core':slot+4,'command':x['cmd'],'started_unix':x['start'],'finished_unix':time.time(),'exit_code':rc}
        if rc==0:
            try:
                m=json.loads((d/'metrics.json').read_text());s=json.loads((d/'settings.json').read_text());logs=[json.loads(l) for l in (d/'train-log.jsonl').read_text().splitlines()]
                assert s['formal']==(not args.smoke) and s['seed']==job['seed'] and s['wd']==job['wd'] and s['fraction']==job['fraction'] and s['max_steps']==job['max_steps'] and s['beta2']==.98 and s['permuted_labels']==job['permuted_labels']
                assert s['source_hashes']=={n:frozen[n] for n in ['grok.py','SPEC.md']}
                assert logs[0]['step']==0 and m['steps_run']==logs[-1]['step']
                assert [l['step'] for l in logs]==list(range(0,m['steps_run']+1,1 if args.smoke else 100))
                expected_stop=min(job['max_steps'],m['grok_step']+1000) if m['grok_step'] is not None else job['max_steps']
                assert m['steps_run']==expected_stop
                for stage,h in m['checkpoint_sha256'].items():
                    if h is not None:assert sha(d/f'checkpoint-{stage}.npz')==h
                assert len(m['ablations']['remove_random_other5'])==10 and len(m['ablations']['keep_only_random_other5'])==10
                rec.update(metrics_sha256=sha(d/'metrics.json'),settings_sha256=sha(d/'settings.json'),steps_run=m['steps_run'],memorization_step=m['memorization_step'],grok_step=m['grok_step'],final_test_accuracy=m['final_log_entry']['test_acc'],top5_share=m['fourier']['final']['share'])
            except Exception as exc:failed=True;rec['verification_failure']=repr(exc)
        else:failed=True
        state['completed'].append(rec);del active[slot];print('FINISH',job['name'],rc,rec.get('grok_step'),rec.get('final_test_accuracy'),flush=True)
    state['active']=[{'name':x['job']['name'],'pid':x['p'].pid,'core':slot+4,'rss_mb':rss(x['p'].pid)} for slot,x in active.items()];state['queued']=len(jobs);state['updated_utc']=utc();(out/'manager-status.json').write_text(json.dumps(state,indent=2)+'\n')
    if failed and not active:break
    if active:time.sleep(1)
state['status']='FAILED' if failed else 'COMPLETE';state['finished_utc']=utc();(out/'manager-status.json').write_text(json.dumps(state,indent=2)+'\n');print(state['status'],'completed',len(state['completed']),'peakRSSMB',state['max_combined_worker_rss_mb'],flush=True);raise SystemExit(1 if failed else 0)
