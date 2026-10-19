"""Regenerate evidence PNGs exclusively from saved numeric artifacts."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':160})


def sparse_figure(artifacts,evidence):
    with np.load(artifacts/'synthetic.npz') as z:
        p=z['rates'];err=z['relative_error'];counts=(err<.001).sum(axis=1)
        fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        axs[0,0].plot(p,counts/err.shape[1],'o-',color='#126a80')
        for x,c in zip(p,counts):axs[0,0].annotate(f'{c}/{err.shape[1]}',(x,c/err.shape[1]),xytext=(0,8),textcoords='offset points',ha='center')
        axs[0,0].set(xlabel='Observation probability p',ylabel='Recovery frequency',ylim=(-.05,1.16),title='Success: relative signal error < 0.001')
        quant=np.quantile(err,[.1,.5,.9],axis=1)
        axs[0,1].fill_between(p,np.maximum(quant[0],1e-14),np.maximum(quant[2],1e-14),alpha=.2,color='#126a80',label='10-90% range')
        axs[0,1].semilogy(p,np.maximum(quant[1],1e-14),'o-',color='#126a80',label='Median')
        axs[0,1].axhline(.001,ls='--',color='#c2632b',label='Success threshold')
        axs[0,1].set(xlabel='Observation probability p',ylabel='Relative L2 error',title='Display floor 1e-14; raw errors unchanged')
        axs[0,1].legend(fontsize=8)
        for ax,idx in zip(axs[1],[0,2]):
            trial=int(np.argsort(err[idx],kind='stable')[12])
            x=np.arange(z['truth'].shape[-1]);m=z['masks'][idx,trial]
            ax.plot(x,z['truth'][idx,trial],color='#333333',lw=1.3,label='Truth')
            ax.plot(x,z['reconstruction'][idx,trial],color='#ba552c',lw=1,ls='--',label='Reconstruction')
            ax.scatter(x[m],z['observations'][idx,trial,m],s=17,color='#0086a5',zorder=3,label='Observed')
            ax.set(xlabel='Signal position',ylabel='Signal value',title=f'p={p[idx]:.1f}, sorted rank 13, trial {trial+1}; error={err[idx,trial]:.3g}')
            ax.legend(fontsize=8,ncol=3)
        fig.suptitle('Sparse recovery: five coefficients, 128 values, seed 7101',fontsize=14)
        fig.savefig(evidence/'sparse-recovery.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=['1','2','3','all'],required=True)
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--evidence',type=Path,default=Path('evidence'))
    a=p.parse_args();a.evidence.mkdir(parents=True,exist_ok=True)
    if a.part in ['1','all']:sparse_figure(a.artifacts,a.evidence)


if __name__=='__main__':main()
