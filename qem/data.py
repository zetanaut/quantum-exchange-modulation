"""Independent physical state generation and directional dephasing target.

Density matrices are simulator input objects, not classically revealed features
for the quantum learners. Hidden preparation parameters never enter a model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULI = np.stack([I, X, Y, Z])
STRINGS = np.stack([np.kron(a, b) for a in PAULI for b in PAULI])
LABELS = [a + b for a in "IXYZ" for b in "IXYZ"]
I4 = np.eye(4, dtype=complex)


def hermitize(rho):
    rho = (rho + rho.conj().T) / 2
    return rho / np.trace(rho).real


def random_state(rng, index):
    """Three preparation families, independent of the learner circuit."""
    family = index % 3
    if family == 0:
        # Gibbs state of two coupled spins, including noncommuting local fields.
        coeff = rng.normal(0, 0.8, 9)
        ops = [STRINGS[k] for k in [5, 10, 15, 4, 8, 12, 1, 2, 3]]
        H = sum(c * op for c, op in zip(coeff, ops))
        beta = rng.uniform(0.15, 3.2)
        ev, U = np.linalg.eigh(H)
        weights = np.exp(-beta * (ev - ev.min()))
        rho = (U * (weights / weights.sum())) @ U.conj().T
        name = "coupled_spin_gibbs"
    elif family == 1:
        # Random local polarizations, followed by independently sampled spin drive.
        local = []
        for _ in range(2):
            v = rng.normal(size=3)
            v *= rng.uniform(0.25, 1.0) / np.linalg.norm(v)
            local.append((I + np.einsum('a,aij->ij', v, PAULI[1:])) / 2)
        rho = np.kron(*local)
        H = sum(c * STRINGS[k] for c, k in zip(rng.normal(size=7), [5, 6, 9, 10, 15, 4, 3]))
        U = expm(-1j * rng.uniform(0.1, 1.4) * H)
        rho = U @ rho @ U.conj().T
        name = "driven_product_mixture"
    else:
        # Randomly oriented rank-one or rank-two preparations with isotropic noise.
        rank = 1 if rng.random() < 0.6 else 2
        G = rng.normal(size=(4, rank)) + 1j * rng.normal(size=(4, rank))
        rho = G @ G.conj().T
        rho /= np.trace(rho).real
        name = "low_rank_noisy_preparation"
    p = rng.uniform(0.0, 0.28)
    return hermitize((1 - p) * rho + p * I4 / 4), name


def directions(rng, count, heldout):
    """A spherical cap n_x > 0.55 is held out in its entirety."""
    rows = []
    while len(rows) < count:
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        if (n[0] > 0.55) == heldout:
            rows.append(n)
    return np.stack(rows)


def observable(n):
    first = np.einsum('...a,aij->...ij', n, PAULI[1:])
    return np.einsum('...ij,kl->...ikjl', first, I).reshape(n.shape[:-1] + (4, 4))


def fragility(rho, P):
    purity = np.einsum('...ij,...ji->...', rho, rho).real
    rp = rho @ P
    return purity - np.einsum('...ij,...ji->...', rp, rp).real


def pauli_features(rho):
    return np.einsum('...ij,kji->...k', rho, STRINGS).real


def gram_features(rho, context):
    # D = 1/2 (Tr G - n^T G n), G_ij = sum_b r_ib r_jb.
    r = pauli_features(rho).reshape((-1, 4, 4))[:, 1:, :]
    G = np.einsum('nab,ncb->nac', r, r)
    diag = 0.5 * (1 - context**2) * np.diagonal(G, axis1=1, axis2=2)
    cross = np.stack([-context[:, a] * context[:, b] * G[:, a, b]
                      for a, b in [(0, 1), (0, 2), (1, 2)]], axis=1)
    return np.concatenate([diag, cross], axis=1)


def make_benchmark(destination, seed=20261009):
    rng = np.random.default_rng(seed)
    states, families = zip(*(random_state(rng, i) for i in range(112)))
    states = np.stack(states)
    outside = directions(rng, 24, False)
    cap = directions(rng, 12, True)
    out = {}
    specs = {
        'train': (np.arange(64), outside[:18], 5, 0),
        'validation': (np.arange(64, 80), outside[18:], 6, 18),
        'test_state': (np.arange(80, 112), outside[18:], 6, 18),
        'test_context': (np.arange(0, 32), cap, 4, 24),
        'test_joint': (np.arange(80, 112), cap, 4, 24),
    }
    manifest = {'seed': seed, 'context_holdout': 'entire spherical cap n_x > 0.55',
                'target': 'Tr(rho^2)-Tr(rho P rho P)',
                'context': 'n=(sin(theta)cos(phi),sin(theta)sin(phi),cos(theta))',
                'source_families': sorted(set(families)),
                'density_matrices': 'simulator states; preparation parameters excluded from learners',
                'splits': {}}
    for split, (ids, pool, per_state, id_offset) in specs.items():
        state_ids, context_ids = [], []
        for idx in ids:
            selected = rng.choice(len(pool), per_state, replace=False)
            state_ids.extend([idx] * per_state)
            context_ids.extend(selected)
        sr = states[np.array(state_ids)]
        c = pool[np.array(context_ids)]
        P = observable(c)
        values = {'rho': sr, 'context': c,
                  'angles': np.stack([np.arccos(c[:, 2]), np.arctan2(c[:, 1], c[:, 0])], axis=1),
                  'P': P, 'y': fragility(sr, P),
                  'state_id': np.array(state_ids), 'context_id': np.array(context_ids) + id_offset,
                  'family': np.array([families[i] for i in state_ids])}
        for k, v in values.items():
            out[f'{split}_{k}'] = v
        manifest['splits'][split] = {'examples': len(sr), 'states': len(ids),
            'context_pool': len(pool), 'y_min': float(values['y'].min()),
            'y_max': float(values['y'].max()), 'y_mean': float(values['y'].mean())}
    plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
    t = np.linspace(0, 1, 51)
    out['bell_t'] = t
    out['bell_rho'] = t[:, None, None] * np.outer(plus, plus) + (1-t[:, None, None]) * np.outer(minus, minus)
    out['bell_context'] = np.tile([0., 0., 1.], (len(t), 1))
    out['bell_angles'] = np.zeros((len(t), 2))
    out['bell_P'] = np.broadcast_to(STRINGS[12], (len(t), 4, 4)).copy()
    out['bell_y'] = (2*t-1)**2
    gram_err = 0.0
    for s in specs:
        gram_err = max(gram_err, float(np.max(np.abs(gram_features(out[s+'_rho'], out[s+'_context']).sum(axis=1)-out[s+'_y']))))
    manifest['gram_identity_max_error'] = gram_err
    assert gram_err < 1e-12
    assert not set(out['train_state_id']) & set(out['test_joint_state_id'])
    assert np.all(out['train_context'][:, 0] <= .55)
    assert np.all(out['test_joint_context'][:, 0] > .55)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **out)
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/benchmark.npz')
    p.add_argument('--seed', type=int, default=20261009)
    args = p.parse_args()
    print(json.dumps(make_benchmark(args.output, args.seed), indent=2))
