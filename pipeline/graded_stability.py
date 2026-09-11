"""Stability of the graded optic lobe: leading eigenvalues of gain * A.

In the linear regime tau da/dt = -a + gain * A a, any eigenvalue with real part > 1
is a mode that grows at rate (Re(lambda) - 1) / tau. A barely unstable mode takes
minutes to grow out of rounding noise and then saturates the whole network.

  python pipeline/graded_stability.py --gain 2.5
"""

import argparse
import argparse as _a

import numpy as np
import scipy.sparse.linalg as sla

from sim_hybrid import Hybrid
from sim_reference import Connectome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=2.5)
    ap.add_argument("--prune", type=float, default=0.005)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tau", type=float, default=20.0)
    args = ap.parse_args()
    ns = _a.Namespace(prune=args.prune, ifscale=1.0, mod_zero=1)
    c = Connectome()
    h = Hybrid(c, ns)
    # Photoreceptors are clamped to the stimulus, so their rows do not evolve.
    A = h.A.astype(np.float64).tolil()
    A[h.eye_g, :] = 0
    A = A.tocsr()
    vals, vecs = sla.eigs(A, k=args.k, which="LR", maxiter=5000, tol=1e-6)
    order = np.argsort(-vals.real)
    tnames = c.types[c.type_id[np.flatnonzero(h.graded)]]
    for i in order:
        lam = vals[i] * args.gain
        growth = (lam.real - 1) / (args.tau / 1000)
        v = np.abs(vecs[:, i])
        top = np.argsort(-v)[:8]
        cells = ", ".join(f"{tnames[j]}" for j in top)
        print(f"lambda(A) {vals[i].real:+.4f}{vals[i].imag:+.4f}i  x gain = {lam.real:+.4f}  "
              f"{'UNSTABLE' if lam.real > 1 else 'stable'} growth {growth:+.2f}/s  top cells: {cells}")
    print(f"max stable gain = {1 / vals.real.max():.3f}")


if __name__ == "__main__":
    main()
