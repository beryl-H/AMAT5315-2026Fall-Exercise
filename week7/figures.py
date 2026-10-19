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


def image_panel(ax,image,title,mask=None):
    cmap=plt.colormaps['gray'].copy();cmap.set_bad('#00a5b5')
    shown=np.ma.array(image,mask=~mask) if mask is not None else image
    ax.imshow(shown,cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
    ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])


def fixed_figures(artifacts,evidence):
    with np.load('inputs/images.npz') as f:images={k:f[k] for k in f.files}
    with np.load(artifacts/'evidence.npz') as f:z={k:f[k] for k in f.files}
    with np.load(artifacts/'case/reconstruction.npz') as f:case=f['reconstruction']
    fig,axs=plt.subplots(4,10,figsize=(16,7),layout='constrained')
    for i in range(10):
        image_panel(axs[0,i],images['validation'][i],f"Validation {i}\nrow {images['validation_ids'][i]}")
        image_panel(axs[1,i],images['test'][i],f"Test {i}\nrow {images['test_ids'][i]}")
        image_panel(axs[2,i],z['observations'][0,i],'Random',z['masks'][0,i])
        image_panel(axs[3,i],z['observations'][1,i],'Central hole',z['masks'][1,i])
    fig.suptitle('All input identities and masks: cyan = missing, black = measured -1',fontsize=14)
    fig.savefig(evidence/'input-images.png');plt.close(fig)
    fig=plt.figure(figsize=(16,10.5),layout='constrained')
    grid=fig.add_gridspec(6,10,height_ratios=[1,1,1,1,1,.85])
    axs=np.array([[fig.add_subplot(grid[r,c]) for c in range(10)] for r in range(5)])
    for i in range(10):
        image_panel(axs[0,i],z['truth'][i],f'Truth {i}')
        for m in range(2):
            base=1+2*m
            image_panel(axs[base,i],z['observations'][m,i],['Random input','Hole input'][m],z['masks'][m,i])
            image_panel(axs[base+1,i],z['fixed'][m,i],f"DCT MSE {z['fixed_mse'][m,i]:.3f}")
    image_panel(fig.add_subplot(grid[5,0]),case,'Standalone case')
    image_panel(fig.add_subplot(grid[5,1]),z['fixed'][0,0],'Batch counterpart')
    note=fig.add_subplot(grid[5,2:]);note.axis('off')
    note.text(.02,.65,f"Standalone vs batch maximum difference: {np.max(np.abs(case-z['fixed'][0,0])):.1g}\nDisplay [-1,1]; raw DCT values are not clipped. Cyan = missing.",fontsize=11,va='center')
    fig.suptitle('Same coefficient budget (k=20), same iterations (100), different missing geometry',fontsize=14)
    fig.savefig(evidence/'fixed-inpainting.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for m,color in enumerate(['#126a80','#ba552c']):
        name=['Random','Central hole'][m];mean=z['fixed_mse'][m].mean()
        ax.plot(range(10),z['fixed_mse'][m],'o-',color=color,label=f'{name}: mean {mean:.5f}')
        ax.axhline(mean,color=color,ls='--',alpha=.6)
    ax.set(xlabel='Digit label (fixed row order)',ylabel='MSE over 400 missing pixels',xticks=range(10),title='DCT errors use raw float64 reconstructions')
    ax.legend();fig.savefig(evidence/'fixed-error.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=['1','2','3','all'],required=True)
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--evidence',type=Path,default=Path('evidence'))
    a=p.parse_args();a.evidence.mkdir(parents=True,exist_ok=True)
    if a.part in ['1','all']:sparse_figure(a.artifacts,a.evidence)
    if a.part in ['2','all']:fixed_figures(a.artifacts,a.evidence)


if __name__=='__main__':main()
