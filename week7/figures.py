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


def diffusion_figures(artifacts,evidence):
    with np.load(artifacts/'evidence.npz') as f:z={k:f[k] for k in f.files}
    with np.load(artifacts/'trace.npz') as f:trace={k:f[k] for k in f.files}
    final_attempt=np.r_[trace['s'][:-1]!=trace['s'][1:],True]
    candidates=np.flatnonzero(final_attempt)
    selected=[candidates[np.argmin(abs(trace['s'][candidates]-target))] for target in [800,500,200,0]]
    fig,axs=plt.subplots(1,5,figsize=(12,3.1),layout='constrained')
    image_panel(axs[0],z['observations'][0,0],'Known observation',z['masks'][0,0])
    for ax,idx in zip(axs[1:],selected):
        delta=np.abs(trace['x_s'][idx][z['masks'][0,0]]-z['observations'][0,0][z['masks'][0,0]]).max()
        image_panel(ax,trace['x_s'][idx],f"s={trace['s'][idx]}, cycle={trace['cycle'][idx]}\nknown max error {delta:.3g}")
    fig.suptitle('Noise-matched known pixels become exact at s=0; all panels [-1,1]',fontsize=13)
    fig.savefig(evidence/'conditioning.png');plt.close(fig)
    averaged=z['diffusion_mse'].mean(axis=2)
    methods=[('DCT',z['fixed_mse'],'#ba552c'),('Diffusion: mean of 4 sample MSEs',averaged,'#126a80'),('Mean completion',z['mean_completion_mse'],'#7b4292')]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for m,ax in enumerate(axs):
        for name,values,color in methods:
            ax.plot(range(10),values[m],'o-',lw=1.3,color=color,label=f'{name}: {values[m].mean():.5f}')
        ax.set(xlabel='Digit label',ylabel='MSE over 400 missing pixels',xticks=range(10),title=['Random mask','Central hole'][m]);ax.legend(fontsize=8)
    fig.suptitle('Frozen Week 6 prior; seeds 7201-7204; own-checkpoint results',fontsize=13)
    fig.savefig(evidence/'recovery-curves.png');plt.close(fig)
    gain=z['fixed_mse']-averaged
    fig,axs=plt.subplots(4,4,figsize=(10,10.5),layout='constrained')
    for row,(m,rule) in enumerate([(0,'largest'),(0,'smallest'),(1,'largest'),(1,'smallest')]):
        i=int(np.argmax(gain[m]) if rule=='largest' else np.argmin(gain[m]))
        image_panel(axs[row,0],z['truth'][i],f"{'Random' if m==0 else 'Hole'}: {rule} gain\ndigit {i}; gain {gain[m,i]:+.4f}")
        image_panel(axs[row,1],z['observations'][m,i],'Observed',z['masks'][m,i])
        image_panel(axs[row,2],z['fixed'][m,i],f"DCT MSE {z['fixed_mse'][m,i]:.4f}")
        image_panel(axs[row,3],z['diffusion'][m,i,0],f"Seed 7201 MSE {z['diffusion_mse'][m,i,0]:.4f}")
    fig.suptitle('Selection uses DCT MSE minus average diffusion MSE, including smallest gains',fontsize=12)
    fig.savefig(evidence/'inpainting-comparison.png');plt.close(fig)
    i=int(np.argmax(z['diversity'][1]));samples=z['diffusion'][1,i].astype(np.float64)
    fig,axs=plt.subplots(2,4,figsize=(11,6),layout='constrained')
    image_panel(axs[0,0],z['truth'][i],f'Truth: digit {i}')
    image_panel(axs[0,1],z['observations'][1,i],'Same observation',z['masks'][1,i])
    for r,ax in enumerate([axs[0,2],axs[0,3],axs[1,0],axs[1,1]]):
        image_panel(ax,samples[r],f"Seed {7201+r}\nMSE {z['diffusion_mse'][1,i,r]:.4f}")
    image_panel(axs[1,2],samples.mean(0),f"Mean completion\nMSE {z['mean_completion_mse'][1,i]:.4f}")
    im=axs[1,3].imshow(samples.var(0,ddof=0),cmap='magma',vmin=0,interpolation='nearest')
    axs[1,3].set_title(f"Population variance\nhidden mean {z['diversity'][1,i]:.4f}",fontsize=9)
    axs[1,3].set_xticks([]);axs[1,3].set_yticks([])
    fig.colorbar(im,ax=axs[1,3],fraction=.046,pad=.04)
    fig.suptitle('Central-hole case with largest missing-pixel population variance',fontsize=14)
    fig.savefig(evidence/'completions.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=['1','2','3','all'],required=True)
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--evidence',type=Path,default=Path('evidence'))
    a=p.parse_args();a.evidence.mkdir(parents=True,exist_ok=True)
    if a.part in ['1','all']:sparse_figure(a.artifacts,a.evidence)
    if a.part in ['2','all']:fixed_figures(a.artifacts,a.evidence)
    if a.part in ['3','all']:diffusion_figures(a.artifacts,a.evidence)


if __name__=='__main__':main()
