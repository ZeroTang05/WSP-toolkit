"""Summarize paired-intervention maps; no claim of task improvement."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    rows=json.loads((a.root/'results.json').read_text());complete=json.loads((a.root/'complete.json').read_text())
    if complete['status']!='real_activation_patching_maps_complete' or complete['requests']!=len(rows):raise ValueError('Incomplete study')
    cases=sorted({r['case'] for r in rows});summaries=[]
    for case in cases:
        selected=[r for r in rows if r['case']==case]
        control=next(r for r in selected if r['positions']=='all')
        null=next(r for r in selected if r['positions']=='last' and r['kind']=='residual' and r['layer']==0)
        endpoint=max((r for r in selected if r['kind']=='residual' and r['positions']=='last'),key=lambda r:r['layer'])
        error=lambda a,b:max(abs(a[k]-b[k]) for k in a)
        if max(error(control['clean'],control['patched']),error(endpoint['clean'],endpoint['patched']),error(null['corrupt'],null['patched']))>1e-5:
            raise ValueError('Analytic control violated')
        mlp=[r for r in selected if r['kind']=='mlp_input'];peak=max(mlp,key=lambda r:r['effects']['logit_difference']['patched_minus_corrupt'])
        worst=min(mlp,key=lambda r:r['effects']['logit_difference']['patched_minus_corrupt'])
        summaries.append(dict(case=case,clean_logit_difference=control['clean']['logit_difference'],
            corrupt_logit_difference=control['corrupt']['logit_difference'],
            strongest_positive_mlp=dict(layer=peak['layer'],**peak['effects']['logit_difference']),
            strongest_negative_mlp=dict(layer=worst['layer'],**worst['effects']['logit_difference']),
            all_embedding_control_max_abs_error=error(control['clean'],control['patched']),
            last_residual_control_max_abs_error=error(endpoint['clean'],endpoint['patched'])))
    result=dict(cases=summaries,results_sha256=hashlib.sha256((a.root/'results.json').read_bytes()).hexdigest(),
        scope='Four authored matched pairs, single-token logit margins. Descriptive causal intervention, not a full circuit, permanent edit or task-accuracy result.')
    (a.root/'analysis.json').write_text(json.dumps(result,indent=2))
    if a.plot:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        fig,axes=plt.subplots(2,1,figsize=(11,5.8),constrained_layout=True)
        for ax,kind,title in zip(axes,['residual','mlp_input'],['Residual stream at the last input position','MLP down-projection input at the last input position']):
            count=max(r['layer'] for r in rows if r['kind']==kind)+1
            data=np.array([[next(r for r in rows if r['case']==case and r['kind']==kind and r['layer']==layer and r['positions']=='last')['effects']['logit_difference']['normalized_gap_change'] for layer in range(count)] for case in cases])
            picture=ax.imshow(data,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.3,vcenter=0,vmax=1))
            ax.set_yticks(range(len(cases)),labels=[s.replace('_',' / ') for s in cases],fontsize=9)
            ax.set_xticks(range(0,count,2));ax.set_xlabel('Residual coordinate (0 = embedding)' if kind=='residual' else 'MLP layer (zero based)')
            ax.set_title(title,fontsize=11)
        fig.colorbar(picture,ax=list(axes),label='(patched - corrupt) / (clean - corrupt), logit difference',shrink=.85)
        fig.suptitle('Paired activation intervention on real Qwen3-0.6B\nLast-residual recovery is an analytic control, not a discovered mechanism',fontsize=12)
        fig.savefig(a.root/'patching-map.png',dpi=180);fig.savefig(a.root/'patching-map.pdf');plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
