#!/usr/bin/env python3
"""Fit FiLM, RBF, affine-state, context-only, and physics-informed baselines."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize
from qem.data import pauli_features, gram_features
from qem.classical_models import FiLMRegressor, metrics, quad_context, squared_distances

SPLITS = ['train','validation','test_state','test_context','test_joint','bell']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/benchmark.npz')
    ap.add_argument('--output',default='results/classical')
    ap.add_argument('--seeds',default='11,23,37')
    ap.add_argument('--iterations',type=int,default=500)
    args=ap.parse_args()
    data=np.load(args.data)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    features={s:pauli_features(data[s+'_rho'])[:,1:] for s in SPLITS}
    contexts={s:data[s+'_context'] for s in SPLITS}
    ys={s:data[s+'_y'] for s in SPLITS}
    result={'training_access':'exact Pauli expectations; no finite-shot training claimed',
            'selection':'validation MSE only; all test splits untouched',
            'models':[]}
    preds={s+'_y':ys[s] for s in SPLITS}

    # Independent finite-difference check of hand-written FiLM gradients.
    film=FiLMRegressor()
    p=film.initial(101)
    f,g=film.value_gradient(p,features['train'][:9],contexts['train'][:9],ys['train'][:9])
    rng=np.random.default_rng(25);direction=rng.normal(size=len(p));direction/=np.linalg.norm(direction)
    eps=1e-5
    fp=film.value_gradient(p+eps*direction,features['train'][:9],contexts['train'][:9],ys['train'][:9])[0]
    fm=film.value_gradient(p-eps*direction,features['train'][:9],contexts['train'][:9],ys['train'][:9])[0]
    error=abs((fp-fm)/(2*eps)-g@direction)
    assert error<1e-7,error
    result['film_gradient_direction_error']=float(error)

    for seed in map(int,args.seeds.split(',')):
        start=time.perf_counter();history=[];best=[np.inf,None,0];calls=[0]
        initial=film.initial(seed)
        def objective(p):
            calls[0]+=1
            return film.value_gradient(p,features['train'],contexts['train'],ys['train'])
        def callback(p):
            val=metrics(ys['validation'],film.predict(p,features['validation'],contexts['validation']))['mse']
            train=metrics(ys['train'],film.predict(p,features['train'],contexts['train']))['mse']
            history.append([len(history)+1,train,val])
            if val<best[0]:best[:]=[val,p.copy(),len(history)]
        callback(initial)
        opt=minimize(objective,initial,jac=True,method='L-BFGS-B',callback=callback,
                     options={'maxiter':args.iterations,'ftol':1e-12,'gtol':1e-8,'maxls':30})
        bp=best[1];row={'model':'classical_film','seed':seed,'parameters':film.n_params,
                       'iterations':int(opt.nit),'best_iteration':best[2],
                       'objective_evaluations':calls[0],'elapsed_seconds':time.perf_counter()-start,
                       'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message),
                       'metrics':{}}
        for s in SPLITS:
            pr=film.predict(bp,features[s],contexts[s]);row['metrics'][s]=metrics(ys[s],pr)
            preds[f'film_{seed}_{s}']=pr
        np.savez_compressed(output/f'film_seed{seed}.npz',parameters=bp,history=np.array(history))
        result['models'].append(row)
        print(json.dumps({'model':'classical_film','seed':seed,'test_joint':row['metrics']['test_joint'],'seconds':row['elapsed_seconds']}),flush=True)

    # Low-dimensional physics-informed quadratic baseline; never call this generic.
    gram={s:gram_features(data[s+'_rho'],contexts[s]) for s in SPLITS}
    coef=np.linalg.lstsq(gram['train'],ys['train'],rcond=None)[0]
    row={'model':'physics_gram_regression','parameters':6,
         'prior':'exact known quadratic response structure; learned linear coefficients',
         'coefficients':coef.tolist(),'metrics':{}}
    for s in SPLITS:
        pr=gram[s]@coef;row['metrics'][s]=metrics(ys[s],pr);preds['gram_'+s]=pr
    result['models'].append(row)
    np.savez_compressed(output/'gram_weights.npz',coefficients=coef)

    for name, feat in [
        ('context_only',{s:quad_context(contexts[s]) for s in SPLITS}),
        ('affine_state',{s:np.einsum('ni,nj->nij',np.column_stack([np.ones(len(features[s])),features[s]]),quad_context(contexts[s])).reshape(len(features[s]),-1) for s in SPLITS})]:
        best=None
        for alpha in [1e-6,1e-3,.1,1.,10.]:
            f=feat['train'];w=np.linalg.solve(f.T@f+alpha*np.eye(f.shape[1]),f.T@ys['train'])
            vm=metrics(ys['validation'],feat['validation']@w)['mse']
            if best is None or vm<best[0]:best=(vm,alpha,w)
        row={'model':name,'parameters':feat['train'].shape[1],'ridge_alpha':best[1],'metrics':{}}
        for s in SPLITS:
            pr=feat[s]@best[2];row['metrics'][s]=metrics(ys[s],pr);preds[name+'_'+s]=pr
        np.savez_compressed(output/(name+'_weights.npz'),coefficients=best[2])
        result['models'].append(row)

    # Nonlinear measured-feature comparator with validation-selected kernel width.
    f={s:np.column_stack([features[s],contexts[s]]) for s in SPLITS}
    train=f['train'];dtrain=squared_distances(train,train)
    dval=squared_distances(f['validation'],train);best=None
    for gamma in [.03,.1,.3,1.]:
        K=np.exp(-gamma*dtrain)
        for alpha in [1e-5,1e-3,.1]:
            w=np.linalg.solve(K+alpha*np.eye(len(K)),ys['train'])
            vm=metrics(ys['validation'],np.exp(-gamma*dval)@w)['mse']
            if best is None or vm<best[0]:best=(vm,gamma,alpha,w)
    row={'model':'rbf_kernel','training_coefficients':len(train),'kernel_gamma':best[1],
         'ridge_alpha':best[2],'metrics':{}}
    for s in SPLITS:
        pr=np.exp(-best[1]*squared_distances(f[s],train))@best[3]
        row['metrics'][s]=metrics(ys[s],pr);preds['rbf_'+s]=pr
    np.savez_compressed(output/'rbf_weights.npz',train_features=train,coefficients=best[3],gamma=best[1])
    result['models'].append(row)
    # Generic quadratic-in-state, quadratic-in-context kernel. This contains the
    # target without inserting its known six-term Gram contraction coefficients.
    rtrain=features['train'];ctrain=contexts['train']
    def quadratic_kernel(s):
        return (1+features[s]@rtrain.T)**2 * (contexts[s]@ctrain.T)**2
    kt=quadratic_kernel('train');kv=quadratic_kernel('validation');best=None
    for alpha in [1e-8,1e-6,1e-4,.01,.1,1.,10.]:
        w=np.linalg.solve(kt+alpha*np.eye(len(kt)),ys['train'])
        vm=metrics(ys['validation'],kv@w)['mse']
        if best is None or vm<best[0]:best=(vm,alpha,w)
    row={'model':'quadratic_product_kernel','training_coefficients':len(rtrain),
         'prior':'degree at most two in Pauli features, degree two in context; no target contraction supplied',
         'ridge_alpha':best[1],'metrics':{}}
    for s in SPLITS:
        pr=quadratic_kernel(s)@best[2]
        row['metrics'][s]=metrics(ys[s],pr);preds['quadratic_kernel_'+s]=pr
    np.savez_compressed(output/'quadratic_kernel_weights.npz',train_features=rtrain,
                        train_context=ctrain,coefficients=best[2])
    result['models'].append(row)
    for row in result['models']:
        if row['model']!='classical_film':
            print(json.dumps({'model':row['model'],'test_joint':row['metrics']['test_joint']}),flush=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'predictions.npz',**preds)


if __name__=='__main__':main()
