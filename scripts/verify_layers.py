#!/usr/bin/env python3
"""Check a trainable, stackable public-API PennyLane/Torch modulation layer."""
from __future__ import annotations
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pennylane as qml
import torch
from qem.layers import ContextExchangeLayer,StreamingExchangeNetwork


def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(109)
    z=np.load('data/benchmark.npz')
    rho=torch.tensor(z['train_rho'][:3],dtype=torch.complex128)
    c=torch.tensor(z['train_context'][:3],dtype=torch.float64)
    layer=ContextExchangeLayer(n_qubits=2,context_dim=3,hidden=8)
    result=layer.self_modulate(rho,c)
    values=layer.gate_parameters(c)
    refs=[]
    for state,ang,a,b in zip(rho,values['angles'],values['a'],values['b']):
        def gates():
            qml.Rot(*ang[:3],wires=0);qml.Rot(*ang[3:6],wires=1)
            qml.IsingZZ(ang[6],wires=[0,1])
        V=qml.matrix(gates,wire_order=[0,1])()
        tau=V@state@V.conj().T
        refs.append(a*state+(1-a)*tau+1j*b*(state@tau-tau@state))
    error=(result-torch.stack(refs)).abs().max().item()
    trace_error=(torch.diagonal(result,dim1=-2,dim2=-1).sum(-1)-1).abs().max().item()
    min_eigenvalue=torch.linalg.eigvalsh(result).min().item()
    assert error<1e-11 and trace_error<1e-11 and min_eigenvalue>-1e-10
    net=StreamingExchangeNetwork(n_qubits=2,context_dim=3,hidden=8,depth=2)
    output=net(rho,c)
    P=torch.tensor(z['train_P'][:3],dtype=torch.complex128)
    expectation=torch.einsum('nij,nji->n',output,P).real
    loss=expectation.square().mean();loss.backward()
    norms=[float(sum((p.grad**2).sum() for p in block.parameters()).sqrt()) for block in net.layers]
    assert all(np.isfinite(norms)) and min(norms)>1e-9
    # Directional finite difference checks that gradients traverse BOTH channels.
    params=list(net.parameters());directions=[torch.randn_like(p) for p in params]
    analytic=sum((p.grad*d).sum() for p,d in zip(params,directions)).item()
    def shifted(sign):
        with torch.no_grad():
            for p,d in zip(params,directions):p.add_(sign*1e-5*d)
        out=net(rho,c)
        value=torch.einsum('nij,nji->n',out,P).real.square().mean().item()
        with torch.no_grad():
            for p,d in zip(params,directions):p.sub_(sign*1e-5*d)
        return value
    numerical=(shifted(1)-shifted(-1))/(2e-5)
    gradient_error=abs(numerical-analytic)
    assert gradient_error<1e-6
    report={'frameworks':{'torch':torch.__version__,'pennylane':qml.__version__},
        'device':'default.mixed, public APIs, unbatched QNodes composed into a batch',
        'state_map_max_abs_error':error,'trace_max_abs_error':trace_error,'minimum_eigenvalue':min_eigenvalue,
        'two_layer_gradient_norms':norms,'directional_gradient_abs_error':gradient_error,
        'streaming_depth':2,'source_copies_per_execution':net.copies_per_execution,
        'status':'PASS','meaning':'Physical state map and differentiable composition; no trained deep-network advantage claimed.'}
    path=Path('results/verification/layer_checks.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
