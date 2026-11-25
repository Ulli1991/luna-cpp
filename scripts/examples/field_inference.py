"""Field-level inference: the whole initial velocity field, optimised so the column density at t_end shows a picture, through
luna-cpp's hand-written adjoint (rollout binary), 3D static (or 2D, the density itself).

After astronomix's field-level inference example (Storcks, Thuerey & Buck 2026, arXiv:2607.18176 sec. 6.3, and its docs notebook:
32^3, 3 x 32^3 = 98 304 velocity controls, Adam lr 0.05, 150 iterations, loss 6.9e-5 -> 3.0e-8). Our setup, in code units: periodic
unit box, rho = 1, P = 1/gamma (c_s = 1), gamma = 5/3; the starting velocity is a random field of wavenumbers 1 <= |k| <= 2
(seeded) at rms Mach 3.5, standing in for their driven turbulence (luna-cpp has no driving); density and pressure stay fixed, the
velocity is the control; t_end = L / (2 v_rms); the column density sum_z rho dz is compared with a target image (a rendered word,
mapped to [1, 1.01] and scaled to the box's mass, as theirs) by the mean squared error. Fixed dt at the rule's dt at the start.
    python field_inference.py <rollout binary> <out.npz> <workdir> [--n=32] [--iters=150] [--lr=0.05] [--dim=3] [--word=LUNA]
"""
import sys
import time

import numpy as np

from rollout_client import Rollout
from blast_emergence import render

GAMMA, MACH = 5.0 / 3.0, 3.5


def velocity(n, dim, seed=7):
    """A random field of 1 <= |k| <= 2, each component, scaled to rms Mach MACH (c_s = 1)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    K = np.sqrt(sum(g ** 2 for g in np.meshgrid(*([k] * dim), indexing="ij")))
    mask = (K >= 1) & (K <= 2)
    v = np.array([np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((n,) * dim)) * mask)) for _ in range(dim)])
    return v * MACH / np.sqrt(np.mean(np.sum(v ** 2, axis=0)))


def main(argv):
    binary, out, work = argv[1:4]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv[4:])
    n, iters, lr, dim, word = int(opts.get("n", 32)), int(opts.get("iters", 150)), float(opts.get("lr", 0.05)), int(opts.get("dim", 3)), opts.get("word", "LUNA")
    N, dV = n ** dim, 1.0 / n ** dim
    P0 = 1.0 / GAMMA
    v0 = velocity(n, dim)
    t_end = 0.5 / MACH

    def U0_of(v):
        U = np.zeros((n,) * dim + (dim + 2,))
        U[..., 0] = 1.0
        for a in range(dim):
            U[..., 1 + a] = v[a]
        U[..., -1] = P0 / (GAMMA - 1) + 0.5 * np.sum(v ** 2, axis=0)
        return U

    probe = Rollout(binary, dim, n, 1, 1e-6, work, tag="probe")
    dt_rule = probe.cfl(U0_of(v0))
    probe.close()
    steps = int(np.ceil(t_end / (0.9 * dt_rule)))
    dt = t_end / steps
    img = render(n, word, 0.0, 1.0)
    # the target: [1, 1.01] scaled to the box's mass, as a column (dim 3) or a density (dim 2)
    tgt = 1.0 + 0.01 * img
    tgt = tgt / tgt.mean()  # the box's mass is 1: the mean column (sum_z rho dz) and the mean density are 1
    r = Rollout(binary, dim, n, steps, dt, work, tag="fli")
    print(f"{dim}D {n}^{dim}: {dim * N} velocity controls, t_end {t_end:.4f}, {steps} steps of dt {dt:.4e} (0.9 of the rule's)")

    def value_grad(v):
        f = r.forward(U0_of(v))
        rho = (f["Q"][:, 0] / f["vol"]).reshape((n,) * dim)
        col = rho.sum(axis=-1) / n if dim == 3 else rho
        d = col - tgt
        L = np.mean(d ** 2)
        colbar = 2.0 * d / d.size
        rhobar = np.broadcast_to(colbar[..., None] / n, rho.shape) if dim == 3 else colbar
        Qbar = np.zeros_like(f["Q"])
        Qbar[:, 0] = rhobar.reshape(N) / f["vol"]
        U0bar, ms = r.adjoint(Qbar)
        g = np.array([U0bar[..., 1 + a] + U0bar[..., -1] * v[a] for a in range(dim)])  # rho = 1
        return L, g, f["ms"], ms, f["violations"]

    v = v0.copy()
    L0, g0, *_ = value_grad(v)
    checks = []
    rng = np.random.default_rng(3)
    for _ in range(3):
        idx = tuple(rng.integers(0, n, dim + 1) % np.array([dim] + [n] * dim))
        h = 1e-6
        vp, vm = v.copy(), v.copy()
        vp[idx] += h
        vm[idx] -= h
        checks.append((g0[idx], (value_grad(vp)[0] - value_grad(vm)[0]) / (2 * h)))
    checks = np.array(checks)
    print(f"L0 {L0:.4e}; gradient against central differences at 3 entries: "
          f"{np.array2string(np.abs(checks[:, 0] - checks[:, 1]) / np.abs(checks[:, 1]), precision=2)} relative")
    m, s = np.zeros_like(v), np.zeros_like(v)
    losses, tf, ta, viol = [], [], [], 0
    best = (np.inf, v)
    t0 = time.perf_counter()
    for it in range(iters):
        L, g, a, b, vi = value_grad(v)
        losses.append(L)
        tf.append(a)
        ta.append(b)
        viol += vi
        if L < best[0]:
            best = (L, v.copy())
        m = 0.9 * m + 0.1 * g
        s = 0.999 * s + 0.001 * g * g
        v = v - lr * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(s / (1 - 0.999 ** (it + 1))) + 1e-8)
        if it % 15 == 0 or it == iters - 1:
            print(f"  it {it:4d}  L = {L:.4e}", flush=True)
    wall = time.perf_counter() - t0
    f = r.forward(U0_of(best[1]))
    rho = (f["Q"][:, 0] / f["vol"]).reshape((n,) * dim)
    peak, _ = r.memory()
    r.close()
    np.savez(out, losses=np.array(losses), v0=v0, v_best=best[1], column=rho.sum(axis=-1) / n if dim == 3 else rho, target=tgt,
             ms_forward=np.array(tf), ms_adjoint=np.array(ta), wall_s=wall, peak_bytes=peak, cfl_violations=viol, checks=checks,
             n=n, dim=dim, steps=steps, dt=dt)
    print(f"wrote {out}: L {losses[0]:.3e} -> best {best[0]:.3e} in {iters} iterations, {wall:.0f} s; forward {np.median(tf):.0f} ms, "
          f"adjoint {np.median(ta):.0f} ms; device peak {peak / 2**20:.0f} MiB ({peak / N:.0f} B/cell); CFL violations {viol}")


if __name__ == "__main__":
    main(sys.argv)
