"""Summarize observed trained-update geometry; never infer task accuracy."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    index=json.loads((root/'cross-model-geometry.json').read_text())
    records=[];inputs={}
    for item in index['models']:
        path=root/item['model']/'geometry.json'
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        data=json.loads(path.read_text());programs={p['program']:p for p in data['programs']}
        descriptions=data['source_descriptions'];tails=[];vshares=[];effective=[]
        for source in descriptions:
            energy=sum(m['frobenius']**2 for m in source['modules'])
            tails.append(sum(m['frobenius']**2 for m in source['modules']
                if m['layer']>=3*(max(v['layer'] for v in source['modules'])+1)//4)/energy)
            vshares.append(sum(m['frobenius']**2 for m in source['modules'] if '.v_proj' in m['module'])/energy)
            effective.append(sum(m['frobenius']**2/energy *
                (sum(s*s for s in m['singular_values'])**2/sum(s**4 for s in m['singular_values']))
                for m in source['modules'] if m['frobenius']>0))
        record=dict(model=item['model'],source_pair_cosines=item['seed_cosines'],mean_cosine=item['mean_seed_cosine'],
            source_top4_energy_mean=item['mean_top4_update_energy'],last_quarter_energy_mean=sum(tails)/len(tails),
            v_projection_energy_mean=sum(vshares)/len(vshares),energy_weighted_module_participation_rank_mean=sum(effective)/len(effective),
            soup_to_seed11_norm=item['soup_norm_to_seed11'],
            equal_norm_soup_attenuation_prediction=((1+2*item['mean_seed_cosine'])/3)**.5,
            soup_parameters=programs['soup']['parameters'],soup_factor_mib=programs['soup']['factor_bytes']/2**20,
            compressed={name:dict(parameters=programs[name]['parameters'],factor_mib=programs[name]['factor_bytes']/2**20,
                retained_weight_energy=(programs[name]['norm']/programs['soup']['norm'])**2,
                storage_ratio=programs[name]['factor_bytes']/programs['soup']['factor_bytes'])
                for name in ['frobenius-r8','frobenius-r4']})
        records.append(record)
    return dict(models=records,input_sha256=inputs,
        scope='Descriptive geometry of 12 trained updates. Three pair cosines share sources; no independent-n=3 inference. SVD energy retention is not accuracy retention. FP32 factor tensor bytes exclude manifests/base.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print('| Model | Cross-seed cosine | Source top-4 energy | Final-quarter layer energy | Soup to r8 energy | Soup to r4 energy | r4 factors MiB |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for r in result['models']:
        c=r['compressed'];print(f"| {r['model']} | {r['mean_cosine']:.3f} | {r['source_top4_energy_mean']:.2%} | {r['last_quarter_energy_mean']:.2%} | {c['frobenius-r8']['retained_weight_energy']:.2%} | {c['frobenius-r4']['retained_weight_energy']:.2%} | {c['frobenius-r4']['factor_mib']:.3f} |")


if __name__=='__main__':main()
