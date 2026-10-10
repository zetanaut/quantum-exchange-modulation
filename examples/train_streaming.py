#!/usr/bin/env python3
"""Train, checkpoint and reload a state-valued quantum predictor with Torch.

Default data are an explicitly synthetic teacher fixture for software
integration. Supply --dataset for your own physics labels and disjoint splits.
"""
from __future__ import annotations
import argparse, copy, csv, io, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qem.predictors import StreamingObservableRegressor, save_predictor, load_predictor


def fixture(seed, config):
    rng = np.random.default_rng(seed+1000)
    torch.manual_seed(seed+2000)
    teacher = StreamingObservableRegressor(**config)
    with torch.no_grad():
        teacher.head.weight.zero_()
        teacher.head.weight[0, 0] = 1.
        teacher.head.bias.zero_()
    d = 2**config["n_qubits"]
    arrays, first_id = {}, 0
    for split, count in [("train", 8), ("validation", 4), ("test", 4)]:
        a = rng.normal(size=(count,d,d))+1j*rng.normal(size=(count,d,d))
        states = a@a.conj().transpose(0,2,1)
        states /= np.trace(states,axis1=1,axis2=2)[:,None,None]
        rho = np.repeat(states,2,axis=0)
        context = rng.uniform(-1,1,(len(rho),config["context_dim"]))
        with torch.no_grad():
            target = teacher(rho,context).numpy()
        arrays.update({f"{split}_rho":rho,f"{split}_context":context,f"{split}_y":target,
                       f"{split}_state_id":np.repeat(np.arange(first_id,first_id+count),2)})
        first_id += count
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--n-qubits",type=int,default=1)
    parser.add_argument("--context-dim",type=int,default=3)
    parser.add_argument("--depth",type=int,default=2)
    parser.add_argument("--hidden",type=int,default=8)
    parser.add_argument("--observables",default=None,help="Comma-separated Pauli words, e.g. ZI,IZ,XX")
    parser.add_argument("--steps",type=int,default=40)
    parser.add_argument("--lr",type=float,default=.02)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path,default=Path("results/streaming_demo"))
    args = parser.parse_args()
    if args.steps < 0 or args.lr <= 0:
        parser.error("steps must be nonnegative and lr must be positive")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    words=args.observables.split(",") if args.observables else [c+"I"*(args.n_qubits-1) for c in "XYZ"]
    config=dict(n_qubits=args.n_qubits,context_dim=args.context_dim,hidden=args.hidden,
                depth=args.depth,observables=words,output_dim=1)
    data=dict(np.load(args.dataset,allow_pickle=False)) if args.dataset else fixture(args.seed,config)
    batches={}
    for split in ["train","validation","test"]:
        rho, context, target=(data[f"{split}_{k}"] for k in ["rho","context","y"])
        if target.ndim==1: target=target[:,None]
        if rho.shape!=(len(target),2**args.n_qubits,2**args.n_qubits):
            raise ValueError(f"Bad {split} density-matrix shape")
        if context.shape!=(len(target),args.context_dim) or target.shape[1:]!=(1,):
            raise ValueError(f"Bad {split} context or target shape")
        batches[split]=(torch.tensor(rho,dtype=torch.complex128),torch.tensor(context),torch.tensor(target))
    if all(f"{s}_state_id" in data for s in batches):
        ids={s:set(data[f"{s}_state_id"].tolist()) for s in batches}
        if any(ids[a]&ids[b] for a,b in [("train","validation"),("train","test"),("validation","test")]):
            raise ValueError("Physical state IDs must be disjoint across splits.")
    torch.manual_seed(args.seed)
    model=StreamingObservableRegressor(**config)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/"dataset.npz",**data)
    history=[]
    best=float("inf")
    best_state=None
    best_step=0
    start=time.perf_counter()
    for step in range(args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        rho,c,y=batches["train"]
        loss=((model(rho,c)-y)**2).mean()
        with torch.no_grad():
            vr,vc,vy=batches["validation"]
            val=float(((model(vr,vc)-vy)**2).mean())
        history.append(dict(step=step,train_mse=float(loss.detach()),validation_mse=val))
        if val<best:
            best,best_step=val,step
            best_state=copy.deepcopy(model.state_dict())
        if step<args.steps:
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite training gradient")
            optimizer.step()
    model.load_state_dict(best_state)
    save_predictor(args.output/"weights.pt",model)
    restored=load_predictor(args.output/"weights.pt")
    scores={}
    predictions={}
    with torch.no_grad():
        for split,(rho,c,y) in batches.items():
            prediction=model(rho,c)
            scores[split]=float(((prediction-y)**2).mean())
            predictions[split]=prediction.numpy()
        r,c,_=batches["test"]
        reload_error=float(torch.max(torch.abs(restored(r,c)-model(r,c))))
    assert reload_error<1e-12
    np.savez(args.output/"predictions.npz",**predictions)
    report=dict(purpose="User-supplied physics dataset" if args.dataset else
                "Synthetic teacher fixture: supervised software integration, not an independent advantage study",
                seed=args.seed,config=config,steps=args.steps,learning_rate=args.lr,
                selected_step=best_step,selection="lowest validation MSE; test evaluated after selection",
                parameters=sum(p.numel() for p in model.parameters()),
                source_copies_per_execution=model.copies_per_execution,
                measurement_settings_note=(f"The script evaluates {len(words)} exact Pauli expectations. "
                    "Group compatible observables for hardware and repeat the full preparation cost for each setting."),
                example_counts={s:len(v[0]) for s,v in batches.items()},
                initial_train_mse=history[0]["train_mse"],metrics=scores,
                checkpoint_reload_max_error=reload_error,seconds=time.perf_counter()-start,
                framework=dict(torch=torch.__version__))
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output/"history.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    fig,ax=plt.subplots(figsize=(7.5,4.3),layout="constrained")
    for key,color,label in [("train_mse","#245b85","Train"),("validation_mse","#a26a21","Validation")]:
        ax.semilogy([r["step"] for r in history],[r[key] for r in history],color=color,label=label)
    ax.axvline(best_step,color="#68747f",ls=":",label="Selected checkpoint")
    ax.set(xlabel="Adam update",ylabel="Mean squared error",title=f"{args.depth}-layer streaming training example")
    ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.2);ax.legend(frameon=False)
    caption=("User-supplied dataset; one configured model and seed, without a comparative baseline." if args.dataset else
             "Synthetic teacher fixture; this plot is not a comparative learning-advantage result.")
    fig.text(.01,-.04,caption,fontsize=9)
    for ext in ["png","pdf","svg"]:
        b=io.BytesIO();fig.savefig(b,format=ext,bbox_inches="tight",dpi=180)
        (args.output/f"streaming_training.{ext}").write_bytes(b.getvalue())
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
