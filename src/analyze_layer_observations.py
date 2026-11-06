"""Recompute descriptive cross-model observation/intervention relationships.

No layer-level p-values: layers are dependent, and these are development proxies,
not generation accuracy. Optional matplotlib produces standalone scientific plots.
"""
import argparse
import hashlib
import json
from pathlib import Path

MODELS=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct']


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):result[order[k]]=(i+j-1)/2
        i=j
    return result


def spearman(first,second):
    a=ranks(first);b=ranks(second);ma=sum(a)/len(a);mb=sum(b)/len(b)
    a=[x-ma for x in a];b=[x-mb for x in b]
    denominator=(sum(x*x for x in a)*sum(x*x for x in b))**.5
    return sum(x*y for x,y in zip(a,b))/denominator if denominator else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true');a=p.parse_args();records=[];sources={};all_rows=[]
    for model in MODELS:
        folder=a.root/model/'probe'
        data={name:(folder/name).read_bytes() for name in ['complete.json','protocol.json','layer_responses.jsonl','layer_programs.json']}
        rows=[json.loads(line) for line in data['layer_responses.jsonl'].decode().splitlines()]
        done=json.loads(data['complete.json']);protocol=json.loads(data['protocol.json'])
        if done['status']!='programming_observations_complete' or sorted(r['layer'] for r in rows)!=list(range(done['layers'])):
            raise ValueError('Incomplete layer intervention data')
        for name,contents in data.items():sources[model+'/'+name]=hashlib.sha256(contents).hexdigest()
        rho=spearman([r['block_influence'] for r in rows],[r['domain_risk_delta'] for r in rows])
        guard=spearman([r['block_influence'] for r in rows],[r['guard_ce_delta'] for r in rows])
        if abs(rho-done['bi_vs_domain_damage_spearman'])>1e-10:raise ValueError('Correlation summary mismatch')
        records.append(dict(model=model,layers=len(rows),domain_spearman=rho,guard_spearman=guard,
            baseline_domain_bits_per_reference_byte=done['baseline_domain_risk'],
            negative_domain_delta_layers=sum(r['domain_risk_delta']<0 for r in rows),
            calibration_examples=len(protocol['calibration_ids']),development_examples=len(protocol['development_ids']),
            guard_examples=len(protocol['guard_ids']),
            lowest_bi_layers=sorted(rows,key=lambda r:r['block_influence'])[:3]))
        all_rows.append(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(models=records,input_sha256=sources,
        scope='Descriptive layer-level association. Dependent layers; no significance inference or task-accuracy claim. Damage is GSM reference bits/byte change or ARC normalized-choice CE change.'),indent=2))
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for ax,record,rows in zip(axes.flat,records,all_rows):
            points=ax.scatter([r['block_influence'] for r in rows],[r['domain_risk_delta'] for r in rows],
                c=[r['normalized_depth'] for r in rows],cmap='viridis',s=35,vmin=0,vmax=1,edgecolors='white',linewidths=.4)
            ax.axhline(0,color='#777777',lw=.7);ax.set_yscale('symlog',linthresh=.05)
            ax.set_ylim(-.35,10);ax.set_xlim(0,1.02)
            ticks=[-.2,-.05,0,.05,.2,1,5]
            ax.set_yticks(ticks,labels=[str(x) for x in ticks]);ax.minorticks_off()
            ax.set_xlabel('Block influence (1 - cosine)');ax.set_ylabel('Deletion damage (reference bits/byte)')
            ax.set_title(record['model']+'; Spearman rho = '+f'{record["domain_spearman"]:.3f}',fontsize=10)
            for row,offset in zip(record['lowest_bi_layers'],[(5,7),(5,-10),(15,2)]):
                ax.annotate(str(row['layer']),(row['block_influence'],row['domain_risk_delta']),xytext=offset,textcoords='offset points',fontsize=7)
            ax.grid(alpha=.16)
        fig.colorbar(points,ax=list(axes.flat),label='Normalized layer depth',shrink=.7)
        fig.suptitle('Observed similarity does not consistently rank deletion damage\nDevelopment proxies; each point is one dependent layer',fontsize=12)
        fig.savefig(a.output.with_suffix('.png'),dpi=180);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
