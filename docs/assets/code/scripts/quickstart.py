#!/usr/bin/env python3
"""Run a small end-to-end Torch optimization through two quantum modulation layers."""
from __future__ import annotations
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from qem.layers import ContextExchangeLayer,StreamingExchangeNetwork


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data=np.load(Path(__file__).resolve().parents[1]/'data/benchmark.npz')
    rho=torch.tensor(data['train_rho'][:4],dtype=torch.complex128)
    context=torch.tensor(data['train_context'][:4])
    observable=torch.tensor(data['train_P'][:4],dtype=torch.complex128)
    # A tiny teacher fixture tests the training path, not scientific generalization.
    teacher=StreamingExchangeNetwork(n_qubits=2,context_dim=3,hidden=8,depth=2)
    with torch.no_grad():
        target=torch.einsum('bij,bji->b',teacher(rho,context),observable).real
    student=StreamingExchangeNetwork(n_qubits=2,context_dim=3,hidden=8,depth=2)
    optimizer=torch.optim.Adam(student.parameters(),lr=.02)
    history=[]
    for step in range(16):
        optimizer.zero_grad(set_to_none=True)
        state=student(rho,context)
        prediction=torch.einsum('bij,bji->b',state,observable).real
        loss=torch.mean((prediction-target)**2)
        history.append(float(loss.detach()))
        if step<15:
            loss.backward();optimizer.step()
    print(json.dumps({'purpose':'small training-path demonstration; no holdout or advantage claim',
        'layers':2,'source_copies_per_execution':student.copies_per_execution,
        'training_examples':4,'initial_mse':history[0],'final_mse':history[-1],
        'best_mse':min(history),'history':history},indent=2))


if __name__=='__main__':main()
