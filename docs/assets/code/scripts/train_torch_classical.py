#!/usr/bin/env python3
"""Train Torch FiLM comparators and select width/regularization on validation only."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from qem.classical_models import metrics
from qem.data import pauli_features
from qem.torch_classical import TorchFiLM

SPLITS = ['train', 'validation', 'test_state', 'test_context', 'test_joint', 'bell']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/benchmark.npz')
    ap.add_argument('--output', default='results/torch_classical')
    ap.add_argument('--seeds', default='17,31,73')
    ap.add_argument('--widths', default='12,24')
    ap.add_argument('--l2', default='0.00001,0.001,0.01')
    ap.add_argument('--steps', type=int, default=180)
    args = ap.parse_args()
    torch.set_num_threads(1)
    data = np.load(args.data)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    features = {s: torch.tensor(pauli_features(data[s+'_rho'])[:, 1:], dtype=torch.float64) for s in SPLITS}
    contexts = {s: torch.tensor(data[s+'_context'], dtype=torch.float64) for s in SPLITS}
    ys = {s: torch.tensor(data[s+'_y'], dtype=torch.float64) for s in SPLITS}
    seeds = [int(v) for v in args.seeds.split(',')]
    widths = [int(v) for v in args.widths.split(',')]
    regularizers = [float(v) for v in args.l2.split(',')]
    histories, weights, predictions, runs = {}, {}, {}, []
    # Independent NumPy formulas validate forward values and all Torch gradients.
    check = TorchFiLM(seed=101)
    loss = ((check(features['train'][:9], contexts['train'][:9])-ys['train'][:9])**2).mean()
    loss += 1e-5 * (check.weights**2).sum()
    loss.backward()
    nv, ng = check.reference.value_gradient(check.weights.detach().numpy(), features['train'][:9].numpy(),
                                           contexts['train'][:9].numpy(), ys['train'][:9].numpy())
    graderr = float(np.max(np.abs(check.weights.grad.numpy()-ng)))
    assert abs(loss.item()-nv) < 1e-12 and graderr < 1e-10
    for width in widths:
        for l2 in regularizers:
            for seed in seeds:
                start = time.perf_counter(); model = TorchFiLM(width=width, seed=seed)
                optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1,
                    max_eval=20, history_size=50, tolerance_grad=1e-9,
                    tolerance_change=1e-12, line_search_fn='strong_wolfe')
                key = f'film_w{width}_l2{l2:g}_seed{seed}'
                history = []; best = [float('inf'), None, 0]; evaluations = [0]
                def evaluate(step):
                    with torch.no_grad():
                        tr = ((model(features['train'],contexts['train'])-ys['train'])**2).mean().item()
                        va = ((model(features['validation'],contexts['validation'])-ys['validation'])**2).mean().item()
                        history.append({'step':step, 'train_mse':tr, 'validation_mse':va,
                                        'closure_evaluations':evaluations[0]})
                        if va < best[0]: best[:] = [va, model.weights.detach().clone(), step]
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss = ((model(features['train'],contexts['train'])-ys['train'])**2).mean()
                    loss = loss + l2 * (model.weights**2).sum()
                    loss.backward(); evaluations[0] += 1
                    return loss
                evaluate(0)
                for step in range(1,args.steps+1):
                    optimizer.step(closure); evaluate(step)
                with torch.no_grad(): model.weights.copy_(best[1])
                row = {'key':key, 'seed':seed, 'width':width, 'l2':l2,
                    'parameters':model.n_params, 'optimizer':'torch.optim.LBFGS / strong_wolfe',
                    'steps':args.steps, 'selected_step':best[2], 'closure_evaluations':evaluations[0],
                    'validation_mse':best[0], 'elapsed_seconds':time.perf_counter()-start, 'metrics':{}}
                with torch.no_grad():
                    for s in SPLITS:
                        p = model(features[s],contexts[s]).numpy()
                        row['metrics'][s] = metrics(ys[s].numpy(),p)
                        predictions[key+'__'+s] = p
                weights[key] = model.weights.detach().numpy(); histories[key] = history; runs.append(row)
                print(json.dumps({'model':'torch_film','width':width,'l2':l2,'seed':seed,
                    'validation':best[0],'test_joint':row['metrics']['test_joint']['mse']}),flush=True)
    scores = [{'width':w,'l2':a,'mean_validation_mse':float(np.mean([r['validation_mse'] for r in runs
               if r['width']==w and r['l2']==a]))} for w in widths for a in regularizers]
    config = min(scores,key=lambda r:r['mean_validation_mse'])
    selected_runs = [r for r in runs if r['width']==config['width'] and r['l2']==config['l2']]
    best_run = min(selected_runs,key=lambda r:r['validation_mse'])
    weights['selected'] = weights[best_run['key']]
    final = TorchFiLM(width=best_run['width'])
    with torch.no_grad(): final.weights.copy_(torch.tensor(weights['selected']))
    torch.save({'state_dict':final.state_dict(),'width':final.width,'selected_run':best_run},out/'film.pt')
    report = {'framework':'PyTorch','torch_version':torch.__version__,'dtype':'float64','device':'CPU',
        'training_access':'exact measured Pauli features; finite-shot training not performed',
        'selection':'width and L2 by mean validation MSE across seeds; iterate and displayed seed by validation only',
        'gradient_check_max_abs_error':graderr, 'config_scores':scores, 'selected_config':config,
        'selected_key':best_run['key'],'selected_runs':selected_runs,'runs':runs}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'histories.json').write_text(json.dumps(histories,indent=2)+'\n')
    np.savez_compressed(out/'weights.npz',**weights)
    np.savez_compressed(out/'predictions.npz',**predictions)
    print(json.dumps({'selected_config':config,'selected_key':best_run['key'],
        'test_joint_mse_all_seeds':[r['metrics']['test_joint']['mse'] for r in selected_runs]}),flush=True)


if __name__ == '__main__': main()
