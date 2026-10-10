#!/usr/bin/env python3
"""Measurement-budget analysis of already-trained models; no shot-training claim.

Quantum affine-head MSE and the unbiased task-aware Pauli estimator have exact
finite-shot formulas. Torch FiLM inference is simulated from binomial Pauli
measurements. Every budget counts physical input-state copies per example.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from qem.data import PAULI, I, pauli_features
from qem.torch_classical import TorchFiLM


def square_variance(mu, shots):
    """Variance of ((sum X)^2-N)/(N(N-1)), an unbiased estimator of mu^2."""
    return 4*mu**2*(1-mu**2)/shots + 2*(1-mu**2)**2/(shots*(shots-1))


def orthogonal_features(rho, context):
    """Eight independent observables (u.sigma or v.sigma) tensor (I,X,Y,Z)."""
    out = []
    for state, n in zip(rho,context):
        # Deterministic stable perpendicular basis, determined by known context.
        anchor = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n,anchor); u /= np.linalg.norm(u); v = np.cross(n,u)
        ops = [np.kron(np.einsum('a,aij->ij',axis,PAULI[1:]),b)
               for axis in (u,v) for b in PAULI]
        out.append([np.trace(state@op).real for op in ops])
    return np.asarray(out)


def joint_moments(heads, correlation, weights, bias, flip=0.):
    attenuation = 1-2*flip
    means = heads*attenuation; corr = correlation*attenuation**2
    prediction = means@weights+bias
    variance = (weights**2*(1-means**2)).sum(axis=1)
    variance += 2*weights[0]*weights[1]*(corr-means[:,0]*means[:,1])
    assert np.min(variance)>-1e-8
    return prediction, np.maximum(variance,0.)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/benchmark.npz')
    ap.add_argument('--quantum',default='results/torch_quantum')
    ap.add_argument('--classical',default='results/torch_classical')
    ap.add_argument('--output',default='results/finite_shot')
    ap.add_argument('--copies',default='240,960,3840,15360')
    ap.add_argument('--replicates',type=int,default=512)
    ap.add_argument('--seed',type=int,default=91402)
    args=ap.parse_args(); torch.set_num_threads(1)
    data=np.load(args.data); split='test_joint'; y=data[split+'_y']
    rho=data[split+'_rho']; context=data[split+'_context']; P=data[split+'_P']
    budgets=[int(c) for c in args.copies.split(',')]
    if any(c%120 or c<240 for c in budgets):
        raise ValueError('Use copy budgets divisible by 120 and at least 240.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    qr=json.loads((Path(args.quantum)/'metrics.json').read_text())
    qp=np.load(Path(args.quantum)/'predictions.npz')
    cr=json.loads((Path(args.classical)/'metrics.json').read_text())
    cw=np.load(Path(args.classical)/'weights.npz')
    film=TorchFiLM(width=cr['selected_config']['width'])
    with torch.no_grad():film.weights.copy_(torch.tensor(cw['selected']))
    r=pauli_features(rho)[:,1:]; rng=np.random.default_rng(args.seed)
    mu=orthogonal_features(rho,context)
    assert np.max(np.abs(.5*(mu**2).sum(axis=1)-y))<1e-12
    m=np.einsum('nij,nji->n',rho,P).real
    rows=[]; bias_rows=[]
    models={}
    for name,info in qr['models'].items():
        seed=info['selected_seed'];prefix=f'{name}__seed{seed}__{split}__'
        summary=info['architecture']['fitted_summary']
        models[name]=(qp[prefix+'heads'],qp[prefix+'correlation'],
                      np.asarray(summary['head_weights']),float(summary['head_bias']))
    models['analytic_exchange_direct_K']=(np.column_stack([m-y/2,m+y/2]),m*m,np.array([-1.,1.]),0.)
    for budget in budgets:
        for name,(heads,corr,w,bias) in models.items():
            pred,var=joint_moments(heads,corr,w,bias)
            squared_bias=float(np.mean((pred-y)**2)); noise=float(np.mean(var)/(budget//2))
            rows.append({'model':name,'copies_per_example':budget,'method':'exact variance',
                         'expected_mse':squared_bias+noise,'squared_bias':squared_bias,
                         'measurement_variance':noise,'shots_per_joint_setting':budget//2})
        # The separate-branch design discards the correlation between ports and
        # consumes four copies for a pair of independent branch shots.
        oldvar=(8*(1-m*m)-2*y*y)/budget
        rows.append({'model':'separate_branch_exchange','copies_per_example':budget,
                     'method':'exact variance','expected_mse':float(oldvar.mean()),
                     'squared_bias':0.,'measurement_variance':float(oldvar.mean()),
                     'shots_per_branch':budget//4})
        shots=budget//8
        classical_var=.25*square_variance(mu,shots).sum(axis=1)
        rows.append({'model':'physics_pauli_unbiased','copies_per_example':budget,
                     'method':'exact variance of independent U-statistics',
                     'expected_mse':float(classical_var.mean()),'squared_bias':0.,
                     'measurement_variance':float(classical_var.mean()),'shots_per_setting':shots,
                     'independent_settings':8})
        shots=budget//15
        samples=2*rng.binomial(shots,(1+r)/2,size=(args.replicates,*r.shape))/shots-1
        fc=torch.tensor(samples.reshape(-1,15),dtype=torch.float64)
        cc=torch.tensor(np.tile(context,(args.replicates,1)),dtype=torch.float64)
        with torch.no_grad(): pred=film(fc,cc).numpy().reshape(args.replicates,len(y))
        trial_mse=np.mean((pred-y)**2,axis=1)
        se=float(trial_mse.std(ddof=1)/np.sqrt(args.replicates))
        rows.append({'model':'torch_film','copies_per_example':budget,
                     'method':'binomial Monte Carlo after exact-expectation training',
                     'expected_mse':float(trial_mse.mean()),'monte_carlo_standard_error':se,
                     'replicates':args.replicates,'shots_per_setting':shots,'independent_settings':15})
    # Explicitly measurement readout error only: no gate-noise claim.
    budget=3840
    for flip in [0.,.005,.01,.02,.05]:
        for name,(heads,corr,w,bias) in models.items():
            pred,var=joint_moments(heads,corr,w,bias,flip)
            bias_rows.append({'model':name,'readout_bit_flip_probability':flip,
                'copies_per_example':budget,'expected_mse':float(np.mean((pred-y)**2)+np.mean(var)/(budget//2)),
                'squared_bias':float(np.mean((pred-y)**2)), 'error_mitigation':False})
    # One independent joint-outcome Monte Carlo audit of the exact variance.
    heads,corr,w,bias=models['analytic_exchange_direct_K']; index=19; N=480
    outcomes=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    probs=np.array([(1+a*heads[index,0]+b*heads[index,1]+a*b*corr[index])/4 for a,b in outcomes])
    counts=rng.multinomial(N,np.clip(probs,0,1),size=50000)
    estimates=(counts@(outcomes@w))/N+bias
    _,variance=joint_moments(heads,corr,w,bias)
    expected=variance[index]/N
    relative_error=abs(estimates.var(ddof=1)/expected-1)
    assert relative_error < .04
    report={'split':split,'examples':len(y),'seed':args.seed,'rows':rows,'readout_noise_rows':bias_rows,
        'training':'All models were trained on exact expectations. This file evaluates finite-shot inference only.',
        'copy_accounting':'Two copies per jointly measured quantum execution; one copy per independently sampled Pauli setting.',
        'pauli_comparator':'Known target structure with an unbiased squared-mean estimator, eight independent rotated Pauli settings. No optimality claim; grouping can change its cost.',
        'classical_film_comparator':'Fifteen independent nonidentity Pauli settings; no grouping or shot-aware retraining.',
        'direct_K':'Analytic exchange and direct K are the same observable and outcome distribution; displayed once.',
        'readout_noise_interpretation':'Independent flips act on the two terminal port detectors. The direct-K row uses this exchange realization; another physical implementation of K can have different noise.',
        'joint_variance_monte_carlo_audit':{'shots':N,'replicates':50000,'expected_variance':float(expected),
            'empirical_variance':float(estimates.var(ddof=1)),'relative_error':float(relative_error),'passed':True},
        'limits':['No device execution','No finite-shot training','No quantum gate noise','No optimal measurement-allocation claim']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'budgets':budgets,'joint_variance_audit':report['joint_variance_monte_carlo_audit']}),flush=True)


if __name__=='__main__':main()
