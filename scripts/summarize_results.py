#!/usr/bin/env python3
"""Create publication-style figures and a factual report from recorded results."""
from __future__ import annotations
import csv,io,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'joint_exchange':'Exchange modulation','exchange_matched_qnn':'QNN with same exchange',
        'no_commutator':'Commutator ablation','torch_film':'Classical FiLM',
        'physics_gram_regression':'Physics Gram features','quadratic_product_kernel':'Quadratic kernel',
        'analytic_exchange_direct_K':'Analytic exchange / direct K',
        'physics_pauli_unbiased':'Unbiased Pauli estimator','separate_branch_exchange':'Separate exchange branches'}
COLORS={'joint_exchange':'#2163ab','exchange_matched_qnn':'#07867c','no_commutator':'#bd4b45',
        'torch_film':'#b7791f','physics_gram_regression':'#626b73','quadratic_product_kernel':'#805aa4',
        'analytic_exchange_direct_K':'#273b53','physics_pauli_unbiased':'#8c54a1','separate_branch_exchange':'#8e969d'}


def sci(x,digits=2):
    if x==0:return '$0$'
    e=int(np.floor(np.log10(abs(x))));m=x/10**e
    return f'${m:.{digits-1}f}\\times10^{{{e}}}$'


def main():
    root=Path(__file__).resolve().parents[1]
    q=json.loads((root/'results/torch_quantum/metrics.json').read_text())
    c=json.loads((root/'results/torch_classical/metrics.json').read_text())
    classical=json.loads((root/'results/classical/summary.json').read_text())
    shot=json.loads((root/'results/finite_shot/metrics.json').read_text())
    verification=json.loads((root/'verification/theory_results.json').read_text())
    data=np.load(root/'data/benchmark.npz')
    qp=np.load(root/'results/torch_quantum/predictions.npz')
    cp=np.load(root/'results/torch_classical/predictions.npz')
    names=['joint_exchange','exchange_matched_qnn','no_commutator']
    rows=[]
    for name in names:
        info=q['models'][name]
        vals=np.array([r['metrics']['test_joint']['mse'] for r in info['runs']])
        selected=next(r for r in info['runs'] if r['seed']==info['selected_seed'])
        rows.append({'model':name,'label':LABELS[name],'parameters':info['architecture']['trainable_parameters'],
                     'seed_mse':vals.tolist(),'median_mse':float(np.median(vals)),
                     'mean_mse':float(vals.mean()),'std_mse':float(vals.std(ddof=1)),
                     'selected_seed':info['selected_seed'],'selected_mse':selected['metrics']['test_joint']['mse'],
                     'metrics_selected':selected['metrics']})
    vals=np.array([r['metrics']['test_joint']['mse'] for r in c['selected_runs']])
    selected=next(r for r in c['selected_runs'] if r['key']==c['selected_key'])
    rows.append({'model':'torch_film','label':LABELS['torch_film'],'parameters':selected['parameters'],
                 'seed_mse':vals.tolist(),'median_mse':float(np.median(vals)),
                 'mean_mse':float(vals.mean()),'std_mse':float(vals.std(ddof=1)),
                 'selected_seed':selected['seed'],'selected_mse':selected['metrics']['test_joint']['mse'],
                 'metrics_selected':selected['metrics']})
    for name in ['quadratic_product_kernel','physics_gram_regression']:
        r=next(r for r in classical['models'] if r['model']==name)
        mse=r['metrics']['test_joint']['mse']
        rows.append({'model':name,'label':LABELS[name],
                     'parameters':r.get('parameters',r.get('training_coefficients')),
                     'seed_mse':[mse],'median_mse':mse,'selected_mse':mse,
                     'metrics_selected':r['metrics']})
    summary={'primary_framework':q['framework'],'data':'two-qubit directional dephasing; quantum state and spherical-cap context holdouts',
             'test_joint_examples':128,'training_examples':320,'validation_examples':96,
             'copy_access':'Independent repeated preparation of a quantum state',
             'results':rows,'finite_shot_inference':shot['rows'],
             'conclusion':'The physical nonlinear mechanism and differentiability are verified. The same-exchange QNN and known Gram features are strong controls; no significant quantum advantage is established.',
             'deep_layer_status':'Reusable two-layer state-valued module passes physicality and gradient checks; no deep-network benchmark or advantage claim.',
             'limits':['Small two-qubit task','Three optimization seeds','Exact labels and expectation gradients',
                       'Inference-only finite-shot analysis','No hardware or gate-noise experiment',
                       'Matched nonlocal primitives, not routed compiled hardware budgets']}
    (root/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'results/summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','parameters_or_kernel_coefficients','median_joint_mse','selected_joint_mse','seed_joint_mse'])
        for r in rows:writer.writerow([r['model'],r['parameters'],r['median_mse'],r['selected_mse'],json.dumps(r['seed_mse'])])
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10.5,
        'legend.fontsize':10,'figure.dpi':160,'savefig.dpi':220,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.color':'#e0e4e8','grid.linewidth':.5,
        'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})
    figures=root/'results/figures';figures.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        buffer=io.BytesIO()
        fig.savefig(buffer,format='pdf',bbox_inches='tight')
        content=buffer.getvalue()
        assert content.rstrip().endswith(b'%%EOF'), 'Incomplete PDF serialization'
        (figures/(name+'.pdf')).write_bytes(content)
        fig.savefig(figures/(name+'.png'),bbox_inches='tight')
        fig.savefig(figures/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.6),layout='constrained')
    ax=axes[0]
    for i,r in enumerate(rows):
        v=np.maximum(r['seed_mse'],1e-14)
        xx=i+np.linspace(-.1,.1,len(v))
        ax.scatter(xx,v,color=COLORS[r['model']],s=30,zorder=3)
    ax.set_yscale('log');ax.set_ylim(4e-15,.2)
    ax.set_xticks(range(len(rows)),['Exchange','Same-exchange\nQNN','No\ncommutator','FiLM','Quadratic\nkernel','Physics\nGram'],rotation=35,ha='right')
    ax.set_ylabel('MSE on unseen states and contexts');ax.set_title('Held-out error across three seeds')
    ax.text(.02,.025,'Gram error displayed at a 1e−14 floor',transform=ax.transAxes,fontsize=8.5,color='#59636b')
    ax=axes[1]
    for name in names:
        info=q['models'][name];run=next(r for r in info['runs'] if r['seed']==info['selected_seed'])
        h=run['trajectory'];x=[r['iteration'] for r in h]
        ax.plot(x,[r['validation_mse'] for r in h],label=LABELS[name],color=COLORS[name],linewidth=1.8)
    ax.set_yscale('log');ax.set_ylim(1e-13,.3);ax.set_xlabel('Torch L-BFGS step')
    ax.set_ylabel('Validation MSE');ax.set_title('Validation-selected seeds')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.27),frameon=False,fontsize=9,borderaxespad=0)
    save(fig,'learning_results')
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.5),layout='constrained')
    ax=axes[0];y=data['test_joint_y'];ax.plot([0,1],[0,1],color='#808080',linewidth=1,zorder=0)
    for name in ['joint_exchange','exchange_matched_qnn']:
        seed=q['models'][name]['selected_seed'];p=qp[f'{name}__seed{seed}__test_joint__prediction']
        ax.scatter(y,p,s=16,facecolors='none',edgecolors=COLORS[name],alpha=.8,label=LABELS[name])
    ax.set(xlabel='Exact response',ylabel='Predicted response',title='Transfer to unseen states and directions')
    ax.legend(frameon=False,fontsize=9)
    ax=axes[1];t=data['bell_t'];ax.plot(t,data['bell_y'],color='#273b53',linewidth=2.4,label='Exact $(2t-1)^2$')
    for name in names:
        seed=q['models'][name]['selected_seed'];p=qp[f'{name}__seed{seed}__bell__prediction']
        ax.plot(t,p,linestyle='--',color=COLORS[name],linewidth=1.35,label=LABELS[name])
    ax.plot(t,cp[c['selected_key']+'__bell'],color=COLORS['torch_film'],linewidth=1.3,label='Classical FiLM')
    ax.set(xlabel='Bell mixing weight $t$',ylabel='Dephasing response',title='Bell fixture: identical local marginals')
    ax.legend(frameon=False,fontsize=9,loc='upper center')
    save(fig,'generalization')
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.6),layout='constrained')
    ax=axes[0]
    for name in ['joint_exchange','exchange_matched_qnn','analytic_exchange_direct_K','physics_pauli_unbiased','torch_film']:
        subset=[r for r in shot['rows'] if r['model']==name]
        ax.plot([r['copies_per_example'] for r in subset],[r['expected_mse'] for r in subset],
                marker='o',markersize=3.5,color=COLORS[name],label=LABELS[name])
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Input-state copies per example')
    ax.set_ylabel('Expected MSE');ax.set_title('Finite-shot inference after ideal training')
    ax.legend(frameon=False,fontsize=9,loc='upper right',bbox_to_anchor=(1,.86))
    ax=axes[1]
    for name in ['joint_exchange','exchange_matched_qnn','analytic_exchange_direct_K']:
        subset=[r for r in shot['readout_noise_rows'] if r['model']==name]
        ax.plot([100*r['readout_bit_flip_probability'] for r in subset],[r['expected_mse'] for r in subset],
                marker='o',markersize=3.5,color=COLORS[name],label=LABELS[name])
    ax.set_yscale('log');ax.set_xlabel('Independent detector-flip probability (%)')
    ax.set_ylabel('Expected MSE');ax.set_title('Two-port detector noise; 3,840 copies')
    ax.legend(frameon=False,fontsize=9)
    save(fig,'measurement_cost')
    print(json.dumps({'results':[{k:r[k] for k in ['model','median_mse','selected_mse']} for r in rows],
                      'figures':[p.name for p in sorted(figures.glob('*.pdf'))]},indent=2))


if __name__=='__main__':main()
