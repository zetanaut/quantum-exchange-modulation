#!/usr/bin/env python3
"""Count source copies and one logical global-exchange gate construction."""
from __future__ import annotations
import argparse, io, json
from pathlib import Path


def estimate(n_qubits, depth, shots, settings):
    if min(n_qubits,depth,shots,settings)<1:
        raise ValueError("All resource inputs must be positive integers.")
    return dict(qubits_per_register=n_qubits,depth=depth,shots_per_setting=shots,settings=settings,
                source_copies_per_execution=depth+1,
                inference_source_copies=(depth+1)*shots*settings,
                concurrent_data_qubits=2*n_qubits,
                data_plus_clean_exchange_ancilla=2*n_qubits+1,
                fredkin_gates_per_execution=2*n_qubits*depth,
                exchange_cnot_upper_construction=16*n_qubits*depth,
                source_zz_cnot_upper_construction=2*max(n_qubits-1,0)*depth,
                dense_joint_state_bytes=16*16**n_qubits,
                excludes=["state preparation", "routing", "local rotations", "readout compilation",
                          "training and gradient executions", "labels", "error correction"],
                interpretation="Logical accounting, not a measured runtime or a hardware advantage")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-qubits",type=int,default=2)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--shots",type=int,default=1000)
    p.add_argument("--settings",type=int,default=3)
    p.add_argument("--plot",action="store_true")
    p.add_argument("--output",type=Path,default=Path("results/resources"))
    a=p.parse_args();report=estimate(a.n_qubits,a.depth,a.shots,a.settings)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    if a.plot:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        depth=np.arange(1,9)
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout="constrained")
        axes[0].plot(depth,depth+1,"o-",color="#245b85",label="Streaming: L + 1")
        axes[0].plot(depth,2**depth,"s-",color="#a26a21",label="Recursive self-copy: 2^L")
        axes[0].set_yscale("log",base=2)
        axes[0].set(xlabel="Modulation depth L",ylabel="Raw input preparations",title="Preparation cost per execution")
        axes[0].legend(frameon=False)
        for n in [1,2,3]:
            axes[1].plot(depth,16*n*depth,"o-",label=f"{n} qubit(s) per register")
        axes[1].set(xlabel="Streaming depth L",ylabel="CNOTs in the exchange construction",title="Logical upper construction: 16 n L")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.2)
        fig.text(.01,-.045,"Arithmetic comparison of different architectures; excludes rotations, preparation and routing. No learning or runtime claim.",fontsize=9)
        for ext in ["png","pdf","svg"]:
            b=io.BytesIO();fig.savefig(b,format=ext,bbox_inches="tight",dpi=180)
            (a.output/f"resource_scaling.{ext}").write_bytes(b.getvalue())
        plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
