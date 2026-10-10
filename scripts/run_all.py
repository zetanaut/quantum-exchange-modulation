#!/usr/bin/env python3
"""Verify the prototype; optionally regenerate the benchmark and all experiments."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',action='store_true',help='Regenerate data and rerun all reported experiments.')
    ap.add_argument('--reference',action='store_true',help='Also rerun the slower independent NumPy quantum fits.')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    def run(*command):
        print('Running: '+' '.join(command),flush=True)
        subprocess.run([sys.executable,*command],cwd=root,check=True)
    if args.train:run('-m','qem.data','--output','data/benchmark.npz')
    run('verification/verify.py')
    run('verification/verify_pennylane.py')
    run('scripts/verify_layers.py')
    if args.train:
        run('scripts/train_torch.py')
        run('scripts/train_torch_classical.py')
        run('scripts/run_classical_benchmark.py')
        if args.reference:run('scripts/run_quantum_benchmark.py')
        run('scripts/finite_shot_benchmark.py')
        run('scripts/summarize_results.py')


if __name__=='__main__':main()
