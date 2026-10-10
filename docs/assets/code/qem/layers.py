"""Reusable state-valued exchange modulation with PyTorch and PennyLane.

This general plug-in layer uses a nonlinear classical context conditioner to
control quantum gates. The benchmark also supplies a smaller physics-structured
conditioner. Both obey the same exact exchange map.
"""
from __future__ import annotations
import math
import warnings
import torch
from torch import nn
import pennylane as qml


class ContextExchangeLayer(nn.Module):
    """Modulate quantum memory using one fresh source preparation.

    ``forward(memory, source, context)`` returns a density matrix, so it can feed
    another quantum layer. Each hardware execution requires an independent
    memory register and source register; passing the same tensor twice in a
    simulator represents two independently prepared copies, never cloning.

    Args:
        n_qubits: Qubits in each equal-size register.
        context_dim: Number of real classical conditioning variables.
        hidden: Width of the tanh context network.

    Batch evaluation intentionally uses public, unbatched default.mixed QNodes:
    this avoids a batched QubitDensityMatrix issue in PennyLane 0.45.1 and keeps
    derivatives with respect to a preceding layer's density matrix intact.
    The primary benchmark has a separately optimized QNode implementation.
    Call ``torch.set_default_dtype(torch.float64)`` before execution: this pinned
    PennyLane version otherwise downcasts Kraus matrices inside default.mixed.
    """
    def __init__(self, n_qubits=2, context_dim=3, hidden=16):
        super().__init__()
        if n_qubits < 1 or context_dim < 1 or hidden < 1:
            raise ValueError('n_qubits, context_dim, and hidden must be positive.')
        self.n_qubits = int(n_qubits)
        self.context_dim = int(context_dim)
        self.n_rotation_angles = 3*self.n_qubits + max(self.n_qubits-1,0)
        self.conditioner = nn.Sequential(nn.Linear(context_dim,hidden),nn.Tanh(),
                                         nn.Linear(hidden,self.n_rotation_angles+2)).double()
        # A moderate positive commutator gain avoids initialization at b=0.
        with torch.no_grad():
            self.conditioner[-1].weight.mul_(.15)
            self.conditioner[-1].bias.zero_()
            self.conditioner[-1].bias[-1] = -math.log(3.)
        dim=2**self.n_qubits
        index=torch.arange(dim*dim)
        permutation=(index%dim)*dim+index//dim
        swap=torch.zeros((dim*dim,dim*dim),dtype=torch.complex128)
        swap[index,permutation]=1
        self.register_buffer('swap',swap)
        self.register_buffer('identity',torch.eye(dim*dim,dtype=torch.complex128))
        dev=qml.device('default.mixed',wires=2*self.n_qubits)

        @qml.qnode(dev,interface='torch',diff_method='backprop')
        def circuit(joint,angles,eta,chi):
            qml.QubitDensityMatrix(joint,wires=range(2*self.n_qubits))
            for j in range(self.n_qubits):
                qml.Rot(*angles[3*j:3*j+3],wires=self.n_qubits+j)
            for j in range(self.n_qubits-1):
                qml.IsingZZ(angles[3*self.n_qubits+j],wires=[self.n_qubits+j,self.n_qubits+j+1])
            plus=torch.cos(eta)*self.identity-1j*torch.sin(eta)*self.swap
            minus=torch.cos(eta)*self.identity+1j*torch.sin(eta)*self.swap
            kraus=torch.stack([torch.cos(chi/2)*plus,torch.sin(chi/2)*minus])
            # PennyLane's validation takes a detached copy, although the stored
            # Kraus tensors retain their autograd graph (verified independently).
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='To copy construct from a tensor')
                qml.QubitChannel(kraus,wires=range(2*self.n_qubits))
            return qml.density_matrix(wires=range(self.n_qubits))
        self.qnode=circuit

    def gate_parameters(self,context):
        """Return conditional gate angles and physical coefficients without measuring rho."""
        raw=self.conditioner(context)
        angles=math.pi*torch.tanh(raw[...,:self.n_rotation_angles])
        eta=(math.pi/2)*torch.sigmoid(raw[...,-2])
        chi=math.pi*torch.sigmoid(raw[...,-1])
        a=torch.cos(eta)**2
        b=.5*torch.sin(2*eta)*torch.cos(chi)
        return {'angles':angles,'eta':eta,'chi':chi,'a':a,'b':b}

    def forward(self,memory,source,context):
        if torch.get_default_dtype()!=torch.float64:
            raise RuntimeError('Call torch.set_default_dtype(torch.float64) before using this layer. '
                               'PennyLane 0.45.1 default.mixed Kraus execution inherits the Torch default precision.')
        dtype_device={'dtype':torch.complex128,'device':self.swap.device}
        memory=torch.as_tensor(memory,**dtype_device)
        source=torch.as_tensor(source,**dtype_device)
        context=torch.as_tensor(context,dtype=torch.float64,device=self.swap.device)
        unbatched=memory.ndim==2
        if unbatched:
            memory=memory.unsqueeze(0);source=source.unsqueeze(0)
            context=context.unsqueeze(0)
        d=2**self.n_qubits
        if memory.ndim!=3 or memory.shape[-2:]!=(d,d) or source.shape!=memory.shape:
            raise ValueError('memory and source must have equal (batch,d,d) or (d,d) shapes.')
        if context.shape!=(len(memory),self.context_dim):
            raise ValueError('Context must match the state batch and configured context_dim.')
        values=self.gate_parameters(context)
        joint=torch.einsum('nab,ncd->nacbd',memory,source).reshape(len(memory),d*d,d*d)
        result=torch.stack([self.qnode(joint[j],values['angles'][j],values['eta'][j],values['chi'][j])
                            for j in range(len(memory))])
        return result[0] if unbatched else result

    def self_modulate(self,rho,context):
        """Quadratic self-copy map; physically requires two independent rho preparations."""
        return self(rho,rho,context)


class StreamingExchangeNetwork(nn.Module):
    """Keep quantum memory and collide it with a fresh raw input at each layer.

    L layers use L+1 source preparations per execution. Reset/discard each used
    source before the next collision; simulator tensors are reusable numerical
    descriptions, not reusable physical copies.
    """
    def __init__(self,n_qubits=2,context_dim=3,hidden=16,depth=2):
        super().__init__()
        if depth<1:raise ValueError('depth must be positive.')
        self.layers=nn.ModuleList([ContextExchangeLayer(n_qubits,context_dim,hidden) for _ in range(depth)])

    @property
    def copies_per_execution(self):
        return len(self.layers)+1

    def forward(self,rho,context):
        memory=rho
        for layer in self.layers:
            memory=layer(memory,rho,context)
        return memory
