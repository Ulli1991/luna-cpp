"""The Kelvin-Helmholtz twin experiment through luna-cpp's hand-written derivatives (rollout binary), static or moving mesh.

The setup is astronomix's (Storcks, Thuerey & Buck 2026, arXiv:2607.18176, sec. 6.2): rho = 1, a tanh shear layer of jump
Delta V = 1 and thickness delta = 0.02 at y = 1/2, Mach 0.3 (P = 6.67 for gamma = 5/3), 256^2 cells. The control p (5 x 2) seeds
the transverse velocity: v_y = sum_{k=2..6} (a_k cos 2 pi k x + b_k sin 2 pi k x) exp(-((y - 1/2) / w)^2), w = 0.04. The truth is
a random p* at 1 % rms of Delta V; the observation is the velocity field at T = 20 t_g (t_g = delta / Delta V); the loss is
L = 1/2 sum_cells |u - u_obs|^2 dV. What cannot match: their box is open in y, ours periodic, so a second, unperturbed layer
sits at y = 0; their scheme is fifth-order WENO-Z with SSPRK4 at an adaptive dt, ours MUSCL-Hancock at a fixed dt. On the moving
mesh the cells move: the observation and the loss are per cell (Lagrangian).

  1. dL/dp at a cold start: one adjoint run, ten tangent runs, central differences (h = 1e-7);
  2. the inversion: L-BFGS-B on the adjoint's gradient from 4 cold starts (p0 ~ N(0, 0.01)); IC error ||v_y(p) - v_y(p*)|| / ||v_y(p*)||;
  3. times per gradient and the device peak.
    python kh_twin.py <rollout binary> <out.npz> <workdir> [--moving] [--n=256] [--starts=4] [--maxiter=60]
"""
import sys
import time

import numpy as np
from scipy.optimize import minimize

from rollout_client import Rollout, centres

GAMMA, DV, DELTA, MACH, W = 5.0 / 3.0, 1.0, 0.02, 0.3, 0.04
P0 = (DV / MACH) ** 2 / GAMMA  # c_s = Delta V / Mach
KS = np.arange(2, 7)
T_G = DELTA / DV


def base(n):
    x, y = centres(n, 2)
    vx = 0.5 * DV * (np.tanh((y - 0.5) / DELTA) - np.tanh((y - 1.0) / DELTA) - np.tanh(y / DELTA))
    return x, y, vx


def basis(n):
    """(10, n, n): the v_y fields of the ten controls (cos, sin per k)."""
    x, y, _ = base(n)
    env = np.exp(-((y - 0.5) / W) ** 2)
    return np.array([f(2 * np.pi * k * x) * env for k in KS for f in (np.cos, np.sin)])


def U0_of(p, n, B):
    _, _, vx = base(n)
    vy = np.tensordot(p, B, axes=1)
    U = np.zeros((n, n, 4))
    U[..., 0] = 1.0
    U[..., 1] = vx
    U[..., 2] = vy
    U[..., 3] = P0 / (GAMMA - 1) + 0.5 * (vx ** 2 + vy ** 2)
    return U


def p_bar(U0bar, p, B):
    """dL/dp from dL/dU0 (rho = 1: d(rho v_y) = dv_y, dE = v_y dv_y)."""
    vy = np.tensordot(p, B, axes=1)
    return np.tensordot(B, U0bar[..., 2] + U0bar[..., 3] * vy, axes=([1, 2], [0, 1]))


def truth(seed=0):
    p = np.random.default_rng(1000 + seed).standard_normal(10)
    return p * 0.01 * DV / np.sqrt(np.sum(p ** 2) / 2)


def main(argv):
    binary, out, work = argv[1:4]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv[4:])
    moving, n = "moving" in opts, int(opts.get("n", 256))
    n_starts, maxiter = int(opts.get("starts", 4)), int(opts.get("maxiter", 60))
    B = basis(n)
    ps = truth()
    T = 20 * T_G
    # dt: the smaller of the rule's at the truth's IC and Courant 0.4 at the largest signal speed; fixed for the run
    probe = Rollout(binary, 2, n, 1, 1e-6, work, moving=moving, tag="probe")
    dt_rule = probe.cfl(U0_of(ps, n, B))
    probe.close()
    dt_c = 0.4 / n / (np.sqrt(GAMMA * P0) + 0.5 * DV + 0.05)
    steps = int(np.ceil(T / min(dt_rule, dt_c)))
    dt = T / steps
    print(f"{'moving' if moving else 'static'} {n}^2: T = {T}, dt = {dt:.4e} ({steps} steps; rule's dt {dt_rule:.4e}, Courant 0.4 {dt_c:.4e})")
    r = Rollout(binary, 2, n, steps, dt, work, moving=moving, tag="kh")
    V0 = 1.0 / n ** 2
    fin = r.forward(U0_of(ps, n, B))
    u_obs = fin["Q"][:, 1:3] / fin["Q"][:, :1]
    rho_true = fin["Q"][:, 0] / fin["vol"]
    print(f"truth: forward {fin['ms']:.0f} ms, CFL violations {fin['violations']}")

    def loss_bars(Q):
        u = Q[:, 1:3] / Q[:, :1]
        d = u - u_obs
        L = 0.5 * V0 * np.sum(d ** 2)
        Qbar = np.zeros_like(Q)
        Qbar[:, 1:3] = V0 * d / Q[:, :1]
        Qbar[:, 0] = -V0 * np.sum(d * Q[:, 1:3], axis=1) / Q[:, 0] ** 2
        return L, Qbar

    stats = dict(fwd=[], adj=[])

    def value_grad(p):
        f = r.forward(U0_of(p, n, B))
        L, Qbar = loss_bars(f["Q"])
        U0bar, ms = r.adjoint(Qbar)
        stats["fwd"].append(f["ms"])
        stats["adj"].append(ms)
        return L, p_bar(U0bar, p, B)

    res = {}
    # 1. gradient checks at a cold start
    p0 = np.random.default_rng(0).normal(0, 0.01, 10)
    L0, g_adj = value_grad(p0)
    U0 = U0_of(p0, n, B)
    g_tan, ms_tan = np.zeros(10), []
    for k in range(10):
        dU = np.zeros_like(U0)
        vy = np.tensordot(p0, B, axes=1)
        dU[..., 2] = B[k]
        dU[..., 3] = vy * B[k]
        dQ, ms = r.tangent(U0, dU)
        ms_tan.append(ms)
        f = r.forward(U0)
        _, Qbar = loss_bars(f["Q"])
        g_tan[k] = np.sum(Qbar * dQ)
    h = 1e-7
    g_fd = np.zeros(10)
    for k in range(10):
        e = np.zeros(10)
        e[k] = h
        g_fd[k] = (loss_bars(r.forward(U0_of(p0 + e, n, B))["Q"])[0] - loss_bars(r.forward(U0_of(p0 - e, n, B))["Q"])[0]) / (2 * h)
    peak, _ = r.memory()
    res.update(p0=p0, L0=L0, g_adjoint=g_adj, g_tangent=g_tan, g_fd=g_fd, ms_forward=np.median(stats["fwd"]),
               ms_adjoint=np.median(stats["adj"]), ms_tangent=np.median(ms_tan), peak_bytes=peak, steps=steps, dt=dt, n=n,
               moving=moving, p_true=ps)
    sc = np.linalg.norm(g_adj)
    print(f"L(p0) = {L0:.6e}; ||g_adj - g_tan|| / ||g|| = {np.linalg.norm(g_adj - g_tan) / sc:.2e}, "
          f"||g_adj - g_FD|| / ||g|| = {np.linalg.norm(g_adj - g_fd) / sc:.2e}")
    print(f"ms: forward {res['ms_forward']:.0f}, adjoint {res['ms_adjoint']:.0f}, one tangent run {res['ms_tangent']:.0f}; "
          f"device peak {peak / 2**20:.0f} MiB ({peak / n**2:.0f} B/cell, the tape {steps * 32} B/cell of it"
          + (f" + positions {steps * 16}" if moving else "") + ")")
    # 2. the inversion
    hists, finals = [], []
    t0 = time.perf_counter()
    for s in range(n_starts):
        start = np.random.default_rng(s).normal(0, 0.01, 10)
        hist = []

        def f(p):
            L, g = value_grad(p)
            hist.append((L, np.linalg.norm(p - ps) / np.linalg.norm(ps)))
            return L, g

        out_ = minimize(f, start, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-14))
        hists.append(np.array(hist))
        finals.append(out_.x)
        print(f"  start {s}: L {hist[0][0]:.3e} -> {hist[-1][0]:.3e}, IC error {hist[0][1]:.3f} -> {hist[-1][1]:.2e}, "
              f"{len(hist)} gradients", flush=True)
    res["inversion_wall_s"] = time.perf_counter() - t0
    H = np.full((n_starts, max(len(h) for h in hists), 2), np.nan)
    for k, hh in enumerate(hists):
        H[k, :len(hh)] = hh
    res.update(histories=H, finals=np.array(finals))
    fin_rec = r.forward(U0_of(finals[0], n, B))
    res.update(rho_true=rho_true.reshape(n, n), rho_rec=(fin_rec["Q"][:, 0] / fin_rec["vol"]).reshape(n, n),
               x_end=fin["x"], vy0_true=np.tensordot(ps, B, axes=1), vy0_rec=np.tensordot(finals[0], B, axes=1),
               ms_forward_all=np.array(stats["fwd"]), ms_adjoint_all=np.array(stats["adj"]))
    r.close()
    np.savez(out, **res)
    print(f"wrote {out}; inversion {res['inversion_wall_s']:.0f} s")


if __name__ == "__main__":
    main(sys.argv)
