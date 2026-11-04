"""Independent NumPy audit of saved Rust inference evidence (never fabricates runs)."""
from pathlib import Path
import argparse
import json
import math
import itertools
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = list('ATSLBEXD')
CASES = {'prior': {}, 'AD': {0: 0, 7: 0}, 'ADX': {0: 0, 7: 0, 6: 0}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_uai(path):
    tokens = iter(Path(path).read_text().split())
    assert next(tokens) in ('MARKOV', 'BAYES')
    dims = [int(next(tokens)) for _ in range(int(next(tokens)))]
    scopes = [[int(next(tokens)) for _ in range(int(next(tokens)))]
              for _ in range(int(next(tokens)))]
    factors = []
    for scope in scopes:
        count = int(next(tokens))
        assert count == math.prod(dims[v] for v in scope)
        values = np.array([float(next(tokens)) for _ in range(count)], dtype=np.float64)
        assert np.all(np.isfinite(values)) and np.all(values >= 0)
        factors.append((scope, values.reshape([dims[v] for v in scope])))
    assert next(tokens, None) is None
    return dims, factors


def enumerate_asia(path, evidence):
    dims, factors = load_uai(path)
    assert dims == [2] * 8
    # Direct indexing of each independently parsed UAI array; last axis fastest.
    assert factors[1][0] == [1, 0]
    np.testing.assert_array_equal(factors[1][1], [[.05, .01], [.95, .99]])
    weights = np.zeros(dims)
    for x in itertools.product(*(range(d) for d in dims)):
        if all(x[v] == state for v, state in evidence.items()):
            weights[x] = math.prod(table[tuple(x[v] for v in scope)] for scope, table in factors)
    z = float(weights.sum())
    assert z > 0
    p = weights / z
    marginals = [p.sum(axis=tuple(j for j in range(8) if j != v)) for v in range(8)]
    joint = p.sum(axis=(0, 2, 4, 5, 6, 7))
    return {'evidence_mass': z, 'marginals': np.array(marginals), 'joint_tl': joint}


def matrix_ising_checks():
    a, b = np.array([[1., 2.], [3., 4.]]), np.array([[2., 0.], [1., 2.]])
    ab = np.array([[sum(a[i, j] * b[j, k] for j in range(2)) for k in range(2)] for i in range(2)])
    np.testing.assert_allclose(ab, [[4, 4], [10, 8]], atol=1e-12, rtol=0)
    trace = sum(a[i, j] * b[j, i] for i in range(2) for j in range(2))
    spins = (-1, 1)
    z = sum(math.exp(.5 * (s*t+t*u+u*s)) for s, t, u in itertools.product(spins, repeat=3))
    assert abs(trace - 12) < 1e-12 and abs(z - 12.60256209895193) < 1e-12
    return {'AB': ab.tolist(), 'trace': trace, 'ising_Z': z}


def recount(plan):
    """Recount from the tree, ignoring Rust's reported costs and retained axes."""
    dims, inputs, output = plan['dimensions'], plan['input_labels'], set(plan['output_labels'])
    steps, leaves = [], []

    def visit(node):
        # serde externally tagged enum: {Leaf:{tensor_index:...}} / {Node:{args,eins}}.
        if 'Leaf' in node:
            i = node['Leaf']['tensor_index']
            leaves.append(i)
            return i, set(inputs[i]), {i}
        data = node.get('Node', node)
        if 'tensor_index' in data:
            i = data['tensor_index']; leaves.append(i)
            return i, set(inputs[i]), {i}
        children = [visit(n) for n in data['args']]
        assert len(children) == 2
        below = children[0][2] | children[1][2]
        outside = output | {v for i, scope in enumerate(inputs) if i not in below for v in scope}
        union = children[0][1] | children[1][1]
        retained = union & outside
        actual = data['eins']['iy']
        assert len(actual) == len(set(actual)) and set(actual) == retained, 'premature elimination or redundant retention'
        assert [set(a) for a in data['eins']['ixs']] == [c[1] for c in children]
        step = {'id': len(inputs) + len(steps), 'children': [c[0] for c in children],
                'retained_labels': sorted(retained), 'work': math.prod(dims[v] for v in union),
                'entries': math.prod(dims[v] for v in retained), 'union_labels': sorted(union)}
        steps.append(step)
        return step['id'], retained, below

    _, final, _ = visit(plan['tree'])
    assert sorted(leaves) == list(range(len(inputs))) and final == output
    generated = max((s['entries'] for s in steps), default=1)
    largest = max([generated] + [math.prod(dims[v] for v in s) for s in inputs])
    return {'work': sum(s['work'] for s in steps), 'largest_tensor': largest,
            'largest_generated_tensor': generated, 'steps': steps}


def audit_query_plans(directory, asia):
    """Check every open query reuses the scalar topology and its axis-size bound."""
    scalar=read_json(directory/'order.json');source=recount(scalar)
    assert scalar['output_labels']==[]
    run=read_json(directory/'run.json')
    observed={int(v) for v in run['observations']}
    expected={f'query-{q}':[q] for q in run['queries'] if q not in observed}
    if asia:expected['joint-tl']=[1,3]
    found={p.name[:-len('-order.json')] for p in directory.glob('*-order.json')}
    assert found==set(expected),('missing or unexpected open query plans',sorted(found),sorted(expected))
    summary=[]
    for name,output in expected.items():
        p=read_json(directory/(name+'-order.json'));c=recount(p)
        assert p['dimensions']==scalar['dimensions'] and p['input_labels']==scalar['input_labels']
        assert p['output_labels']==output
        root=p['tree'].get('Node',p['tree'])
        assert root['eins']['iy']==output, 'ordered query output differs from requested states'
        multiplier=math.prod(p['dimensions'][v] for v in output)
        assert len(c['steps'])==len(source['steps'])
        for s,q in zip(source['steps'],c['steps']):
            assert s['id']==q['id'] and s['children']==q['children'], 'query changed scalar topology'
            assert set(q['retained_labels'])-set(s['retained_labels']) <= set(output)
            assert set(q['union_labels'])-set(s['union_labels']) <= set(output)
            assert s['work']<=q['work']<=multiplier*s['work'], 'pair-work dimension-product bound violated'
            assert s['entries']<=q['entries']<=multiplier*s['entries'], 'intermediate dimension-product bound violated'
        assert c['work']<=multiplier*source['work']
        assert c['largest_generated_tensor']<=multiplier*source['largest_generated_tensor']
        saved=read_json(directory/(name+'-complexity.json'))
        for key in ('work','largest_tensor','largest_generated_tensor'):assert c[key]==saved[key],(name,key)
        assert len(saved['steps'])==len(c['steps'])
        for a,b in zip(c['steps'],saved['steps']):
            for key in ('id','children','work','entries','union_labels','retained_labels'):
                assert a[key]==b[key],(name,key)
        settings=p['settings']
        assert settings['query_axis_size_multiplier']==multiplier
        assert settings['scalar_work']==source['work']
        assert settings['scalar_largest_generated_tensor']==source['largest_generated_tensor']
        assert settings['scalar_planning_method']==scalar['method']
        assert 'scalar evidence plan' in settings['query_topology_source']
        summary.append({'query':output,'axis_size_multiplier':multiplier,'work':c['work'],
                        'largest_generated_tensor':c['largest_generated_tensor'],
                        'same_scalar_topology':True,'per_step_bounds_passed':True})
    return {'directory':str(directory),'query_count':len(summary),'scalar_work':source['work'],
            'scalar_largest_generated_tensor':source['largest_generated_tensor'],'queries':summary}


def marginals(run):
    data = run.get('marginals', run.get('node_marginals'))
    if isinstance(data, dict):
        data = [data.get(str(v), data.get(NAMES[v] if v < 8 else str(v))) for v in range(len(data))]
    if isinstance(data, list):
        data = [[float('nan'), float('nan')] if v is None else v for v in data]
    return np.asarray(data, dtype=float)


def joint(run):
    return np.asarray(run.get('joint_tl', run.get('joint_TL', run.get('tl_joint'))), dtype=float).reshape(2, 2)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def draw_query(path, model):
    _, factors = load_uai(model)
    # Factor boxes and a junction for each variable make scope/hyperedges explicit.
    positions = {0:(.7, 1.9), 1:(2.4, 1.9), 2:(.7, .1), 3:(2.4, .1),
                 4:(4.1, .1), 5:(4.1, 1.9), 6:(5.8, 1.9), 7:(5.8, .1)}
    junction = {0:(1.0, 3), 1:(3, 3), 2:(1, -1), 3:(3, -1),
                4:(4.7, -1), 5:(4.7, 3), 6:(6.7, 3), 7:(6.7, -1)}
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (scope, _) in enumerate(factors):
        x, y = positions[i]
        ax.add_patch(plt.Rectangle((x-.42,y-.3),.84,.6,facecolor='#dbeafe',edgecolor='#1d4ed8',zorder=4))
        ax.text(x,y, f'f{i}\n'+','.join(NAMES[v] for v in scope),ha='center',va='center',fontsize=9,zorder=5)
        for k,v in enumerate(scope):
            jx,jy=junction[v]
            ax.plot([x+(k-(len(scope)-1)/2)*.2,jx],[y,jy],color='#64748b',lw=1,zorder=1)
    for v,(x,y) in junction.items():
        ax.scatter([x],[y],c='#111827',s=20,zorder=3)
        if v in (0,7):
            ax.text(x,y+.3 if y>0 else y-.3,f'{NAMES[v]} = 0: (1,0)',ha='center',color='#c2410c')
        elif v==1:
            ax.plot([x,x],[y,y+.5],color='#15803d',lw=3)
            ax.text(x,y+.65,'T OPEN',ha='center',color='#15803d')
        else:
            ax.text(x,y+.2 if y>0 else y-.25,NAMES[v]+' summed',ha='center')
    ax.set(xlim=(-.1,7.7),ylim=(-1.9,4.1),title='ASIA AD query: p(T, A=0, D=0); normalize by evidence mass')
    ax.axis('off')
    # PNG uses the identical figure as the formal SVG for local visual inspection.
    plt.tight_layout();plt.savefig(path.with_suffix('.png'),dpi=180)
    savefig(path)


def plot_asia(exact, baseline, directory):
    fig, axes = plt.subplots(1,3,figsize=(12,3.8))
    for ax,(case,run) in zip(axes,exact.items()):
        x=np.arange(3); ref=baseline[case]['marginals'][[1,3,4],0]
        ax.bar(x,ref,color='#93c5fd',label='NumPy enumeration')
        ax.scatter(x,marginals(run)[[1,3,4],0],marker='x',color='#b91c1c',label='Rust contraction')
        ax.set(xticks=x,xticklabels=['TB','Lung cancer','Bronchitis'],ylim=(0,1),title=case,ylabel='P(present | evidence)')
    axes[0].legend(fontsize=8);savefig(directory/'posteriors.png')
    fig,axes=plt.subplots(3,3,figsize=(11,9))
    dependence_limit=max(float(np.max(abs(joint(r)-np.outer(marginals(r)[1],marginals(r)[3])))) for r in exact.values())
    dependence_limit=max(dependence_limit,1e-12)
    for col,(case,run) in enumerate(exact.items()):
        j=joint(run);m=marginals(run);product=np.outer(m[1],m[3]);diff=j-product
        for row,data in enumerate([j,product,diff]):
            ax=axes[row,col];im=ax.imshow(data,cmap='coolwarm' if row==2 else 'Blues',vmin=-dependence_limit if row==2 else 0,vmax=dependence_limit if row==2 else 1)
            for i,k in itertools.product(range(2),repeat=2):
                dark=abs(data[i,k])>.7*dependence_limit if row==2 else data[i,k]>.6
                ax.text(k,i,f'{data[i,k]:.6f}',ha='center',va='center',color='white' if dark else 'black')
            ax.set(title=f'{case}: '+['joint P(T,L)','product P(T) P(L)','dependence: joint - product'][row],xlabel='L state',ylabel='T state',xticks=[0,1],yticks=[0,1])
            fig.colorbar(im,ax=ax,shrink=.7)
    savefig(directory/'joint.png')


def plot_orders(plans, directory, filename):
    fig,axes=plt.subplots(1,len(plans),figsize=(7*len(plans),6),squeeze=False)
    for ax,(label,plan,cost) in zip(axes[0],plans):
        n=len(plan['input_labels'])
        # DFS leaf order makes the saved binary tree planar; original factor IDs
        # remain labels, rather than imposing an ordering that crosses edges.
        children={s['id']:s['children'] for s in cost['steps']}
        leaf_order=[]
        def visit_order(i):
            if i<n:leaf_order.append(i)
            else:
                for child in children[i]:visit_order(child)
        visit_order(cost['steps'][-1]['id'])
        xy={i:(x,0) for x,i in enumerate(leaf_order)}
        largest=max(cost['steps'],key=lambda s:s['entries']) if cost['steps'] else None
        for s in cost['steps']:
            a,b=s['children'];x=(xy[a][0]+xy[b][0])/2;y=max(xy[a][1],xy[b][1])+1
            xy[s['id']]=(x,y)
            for child in (a,b):ax.plot([xy[child][0],x],[xy[child][1],y],color='#64748b',lw=.7)
            highlight=largest is not None and s['id']==largest['id']
            if highlight:
                ax.scatter([x],[y],s=100,facecolor='#fbbf24',edgecolor='#b45309',zorder=5)
            if n<30:
                text=f"{s['id']}: {s['entries']} entries"
                if highlight:text+='\nLARGEST GENERATED\nlabels='+str(s['retained_labels'])
                ax.annotate(text,(x,y),xytext=(0,12 if highlight else 0),textcoords='offset points',fontsize=8 if highlight else 7,ha='center',va='bottom' if highlight else 'center',color='#92400e' if highlight else '#111827',zorder=6,bbox={'facecolor':'#fef3c7' if highlight else 'white','alpha':.95,'edgecolor':'#b45309' if highlight else 'none'})
        if n<30:
            for i,scope in enumerate(plan['input_labels']):ax.text(xy[i][0],-.3,f'f{i}\n{scope}',ha='center',fontsize=7)
        info='' if largest is None else f"\nLargest generated: node {largest['id']}, labels={largest['retained_labels']}"
        if n>=30:info=f"\nLargest generated: node {largest['id']}; {len(largest['retained_labels'])} retained labels (complexity.json)"
        ax.set_title(f"{label}\nwork={cost['work']:,}; largest including inputs={cost['largest_tensor']:,}\nlargest generated={cost['largest_generated_tensor']:,}"+info,fontsize=10 if n>=30 else 12)
        ax.axis('off')
    savefig(directory/filename)
    if filename.startswith('promedus'):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,key,title in zip(axes,['work','largest_generated_tensor'],['Pair contraction work (terms)','Largest generated tensor (entries)']):
            values=[c[key] for _,_,c in plans]
            bars=ax.bar([name for name,_,_ in plans],values,color=['#93c5fd','#86efac'])
            ax.set_yscale('log');ax.set(title=title,ylabel='logarithmic scale')
            for bar,value in zip(bars,values):ax.annotate(f'{value:,}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),textcoords='offset points',ha='center')
            ax.margins(y=.25)
        savefig(directory/'promedus-costs.png')


def plot_bp(cases, directory, filename):
    fig,axes=plt.subplots(2,len(cases),figsize=(5*len(cases),7),squeeze=False)
    for col,(label,run,target,ids) in enumerate(cases):
        history=run.get('history',run.get('trace'))
        assert history, 'BP history missing'
        sweeps=[h['sweep'] for h in history]
        residual=[h.get('residual',h.get('undamped_residual')) for h in history]
        errors=[float(np.max(np.abs(marginals(h)[ids]-target))) for h in history]
        assert all(np.isfinite(residual)) and all(np.isfinite(errors))
        final_error=float(np.max(abs(marginals(run)[ids]-target)))
        assert abs(final_error-errors[-1])<1e-12, 'returned marginals differ from last history state'
        converged=run.get('converged')
        if converged is not None:
            assert bool(converged)==bool(residual[-1]<1e-10), 'convergence flag disagrees with returned residual'
        work=[h['cumulative_update_work'] for h in history]
        assert all(b>=a for a,b in zip(work,work[1:])), 'cumulative work decreased'
        axes[0,col].semilogy(sweeps,np.maximum(residual,1e-18));axes[0,col].axhline(1e-10,color='red',ls='--',label='tolerance')
        axes[0,col].set(title=label+' residual',xlabel='sweep',ylabel='max |F(m)-m|');axes[0,col].legend()
        axes[1,col].semilogy(sweeps,np.maximum(errors,1e-18));axes[1,col].set(title=label+' marginal error',xlabel='sweep',ylabel='max absolute error')
        if filename.startswith('promedus'):
            axes[0,col].set_title('Promedus message residual\n'+('CONVERGED' if converged else 'UNCONVERGED')+f' after {sweeps[-1]} sweeps')
            axes[0,col].text(.03,.08,f'final residual = {residual[-1]:.3e}',transform=axes[0,col].transAxes,fontsize=10)
            axes[1,col].set_title('Promedus query error: IDs 59, 102, 139')
            axes[1,col].text(.35,.85,f'final maximum error = {errors[-1]:.5f}',transform=axes[1,col].transAxes,fontsize=10)
    savefig(directory/filename)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--asia-only',action='store_true',help='Audit ASIA only; explicitly defer Promedus checks')
    args=parser.parse_args();root=args.root;out=root/'evidence';out.mkdir(exist_ok=True)
    report={'scope':'baseline-only' if args.baseline_only else 'asia-only' if args.asia_only else 'full','checks':[],'failures':[]}
    def check(name,fn):
        try:
            value=fn();report['checks'].append({'name':name,'result':value});print('PASS',name);return value
        except Exception as e:
            report['failures'].append({'name':name,'error':str(e),'type':type(e).__name__});print('FAIL',name,str(e),file=sys.stderr);return None
    check('NumPy direct-sum matrix / enumerated Ising reference',matrix_ising_checks)
    baseline={}
    for case,e in CASES.items():
        def baseline_check(case=case,e=e):
            b=enumerate_asia(root/'asia.uai',e);baseline[case]=b
            expected={'prior':1.,'AD':.004501375,'ADX':.00098822675}[case]
            assert abs(b['evidence_mass']-expected)/expected<1e-12
            return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in b.items()}
        check('independent ASIA '+case,baseline_check)
    check('AD query SVG',lambda:(draw_query(out/'graphs.svg',root/'asia.uai'),'evidence/graphs.svg')[1])
    if not args.baseline_only:
        exact={}
        for case in CASES:
            def audit(case=case):
                run=read_json(root/'artifacts'/case/'run.json');b=baseline[case]
                m=marginals(run);j=joint(run)
                err=max(float(np.max(abs(m-b['marginals']))),float(np.max(abs(j-b['joint_tl']))))
                mass=run['evidence_mass'];rel=abs(mass-b['evidence_mass'])/b['evidence_mass']
                assert err<1e-10 and rel<1e-10,(err,rel)
                np.testing.assert_allclose(m.sum(axis=1),1,atol=1e-12,rtol=0)
                assert abs(j.sum()-1)<1e-12
                sanity=run['sanity_checks'];reference=matrix_ising_checks()
                np.testing.assert_allclose(np.asarray(sanity['matrix_ab']).reshape(2,2),reference['AB'],atol=1e-12,rtol=0)
                assert abs(sanity['trace_ab']-reference['trace'])<1e-12
                assert abs(sanity['ising_contraction']-reference['ising_Z'])<1e-12
                assert abs(sanity['ising_enumeration']-reference['ising_Z'])<1e-12
                exact[case]=run
                return {'max_absolute_marginal_error':err,'relative_evidence_mass_error':rel}
            check('Rust ASIA '+case,audit)
        if len(exact)==3:check('ASIA probability plots',lambda:(plot_asia(exact,baseline,out),'posteriors.png, joint.png')[1])
        def plans_audit(paths,filename):
            plans=[]
            for label,path in paths:
                p=read_json(path/'order.json');c=recount(p);saved=read_json(path/'complexity.json')
                for k in ('work','largest_tensor','largest_generated_tensor'):assert c[k]==saved[k],(label,k,c[k],saved[k])
                for a,b in zip(c['steps'],saved['steps']):
                    for k in ('work','entries','retained_labels'):assert a[k]==b[k],(label,k)
                assert len(c['steps'])==len(saved['steps'])
                plans.append((label,p,c))
            plot_orders(plans,out,filename)
            return {label:{k:c[k] for k in ('work','largest_tensor','largest_generated_tensor')} for label,p,c in plans}
        check('ASIA independent plan costs',lambda:plans_audit([(m,root/'runs'/m) for m in ('random','greedy')],'orders.png'))
        def replay_audit():
            reference=baseline['AD'];result={}
            for name in ('random','greedy','replay'):
                r=read_json(root/'runs'/name/'run.json')
                err=float(np.max(abs(marginals(r)-reference['marginals'])))
                mass_error=abs(r['evidence_mass']-reference['evidence_mass'])/reference['evidence_mass']
                assert err<1e-10 and mass_error<1e-10
                result[name]={'max_absolute_marginal_error':err,'relative_evidence_mass_error':mass_error}
            costs=read_json(root/'runs'/'analysis'/'complexity.json')
            original=recount(read_json(root/'runs'/'random'/'order.json'))
            for k in ('work','largest_tensor','largest_generated_tensor'):assert costs[k]==original[k]
            assert not (root/'runs'/'analysis'/'run.json').exists(), 'analysis must not write inference run'
            return result
        check('ASIA random / greedy / replay and analyze-only',replay_audit)
        check('ASIA open query plans / scalar-topology dimension bounds',lambda:[audit_query_plans(p,True) for p in
              [root/'artifacts'/case for case in CASES]+[root/'runs'/name for name in ('random','greedy','replay')]])
        check('ASIA BP residual and error plots',lambda:(plot_bp([(c,read_json(root/'artifacts'/('bp-'+c)/'run.json'),baseline[c]['marginals'],list(range(8))) for c in CASES],out,'bp.png'),'bp.png')[1])
        if not args.asia_only:
            check('Promedus independent plan costs',lambda:plans_audit([(m,root/'runs'/('promedus-'+m)) for m in ('greedy','treesa')],'promedus-orders.png'))
            check('Promedus six open query plans / scalar-topology dimension bounds',lambda:[audit_query_plans(root/'runs'/('promedus-'+m),False) for m in ('greedy','treesa')])
        # Published answers are parsed directly; MAR has state-order values, not yes/no assumptions.
        def benchmark():
            pr=float((root/'promedus'/'Promedus_14.uai.PR').read_text().split()[-1])
            tokens=iter((root/'promedus'/'Promedus_14.uai.MAR').read_text().split());assert next(tokens)=='MAR'
            n=int(next(tokens));mar=[np.array([float(next(tokens)) for _ in range(int(next(tokens)))]) for _ in range(n)]
            ids=[59,102,139];target=np.array([mar[i] for i in ids]);runs=[];values=[]
            for method in ('greedy','treesa'):
                r=read_json(root/'runs'/('promedus-'+method)/'run.json');z=r.get('log10_partition',r.get('log10_evidence_partition_sum'))
                assert z is not None and abs(z-pr)<1e-4,(method,z,pr)
                m=marginals(r);selected=m[ids] if len(m)>139 else m
                assert np.max(abs(selected-target))<1e-5
                runs.append(selected);values.append(z)
            assert abs(values[0]-values[1])<1e-10 and np.max(abs(runs[0]-runs[1]))<1e-10
            bp=read_json(root/'artifacts'/'bp-promedus'/'run.json')
            plot_bp([('Promedus selected queries',bp,target,ids)],out,'promedus-bp.png')
            return {'published_log10Z':pr,'exact_log10Z':values,'selected_state1':target[:,1].tolist(),'bp_converged':bp.get('converged')}
        if not args.asia_only:
            check('Promedus PR / MAR and BP',benchmark)
    # Keep each execution as a distinct report, including failures; never discard prior evidence.
    from datetime import datetime,timezone
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report_path=out/('audit-'+stamp+'.json');report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Audit report:',report_path)
    return 1 if report['failures'] else 0


if __name__=='__main__':
    raise SystemExit(main())
