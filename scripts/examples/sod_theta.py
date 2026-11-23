"""Sod's six parameters theta = (rho_L, P_L, u_L, rho_R, P_R, u_R) through luna-cpp's hand-written derivatives (build/sod_theta).

The problem is astronomix's shock-tube sensitivity setup (Storcks, Thuerey & Buck 2026, arXiv:2607.18176, sec. 5.2): [0, 1], N = 200,
gamma = 5/3, outflow walls, 250 fixed steps of dt = 8e-4 to t = 0.2, J = 1/2 int (rho^2 + v^2 + P^2) dx at t. What it does:
  1. dJ/dtheta at the Sod state by one adjoint run and by six tangent runs;
  2. the finite-difference sweep: h = 1e-2 .. 1e-8 along each parameter, the residual of the gradient against the central and
     against the closer one-sided difference (astronomix's measure), ||g - g_FD|| / ||g||;
  3. the sensitivity map dJ/dU0 per cell (one adjoint), checked against central differences at a few cells;
  4. the inverse problem: recover theta from the final density field rho(x, t) alone (loss L = 1/2 int (rho - rho_obs)^2 dx) by
     L-BFGS-B on the adjoint's gradient, from 8 starts;
  5. times per gradient.
Writes everything to one npz. Usage (on a GPU node; python with numpy and scipy):
    python sod_theta.py <sod_theta binary> <out.npz> [--limiter=minmod]
"""
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize

THETA_SOD = np.array([1.0, 1.0, 0.0, 0.125, 0.1, 0.0])
NAMES = ["rho_L", "P_L", "u_L", "rho_R", "P_R", "u_R"]
HS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


class Tube:
    """The sod_theta binary in serve mode: one command a line, one answer a line."""

    def __init__(self, binary, *flags):
        self.p = subprocess.Popen([binary, *flags], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def ask(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().split()
        if not out or out[0] == "error":
            raise RuntimeError(f"sod_theta: {line} -> {out}")
        return out

    @staticmethod
    def th(theta):
        return " ".join(repr(float(t)) for t in theta)

    def J(self, theta):
        out = self.ask("forward " + self.th(theta))
        return float(out[1]), float(out[3])

    def adjoint(self, which, theta):
        out = [float(v) for v in self.ask(f"adjoint {which} " + self.th(theta))]
        return out[0], np.array(out[1:7]), out[7], out[8]

    def tangent(self, which, theta):
        out = [float(v) for v in self.ask(f"tangent {which} " + self.th(theta))]
        return out[0], np.array(out[1:7]), out[7]

    def close(self):
        self.p.stdin.write("quit\n")
        self.p.stdin.flush()
        self.p.wait()


def fd_sweep(J, theta, g):
    """Central and closer-one-sided residuals of g against differences of J along each parameter."""
    J0 = J(theta)
    central, onesided = [], []
    for h in HS:
        gc, go = np.zeros(6), np.zeros(6)
        for k in range(6):
            e = np.zeros(6)
            e[k] = h
            Jp, Jm = J(theta + e), J(theta - e)
            gc[k] = (Jp - Jm) / (2 * h)
            fwd, bwd = (Jp - J0) / h, (J0 - Jm) / h
            go[k] = fwd if abs(fwd - g[k]) < abs(bwd - g[k]) else bwd
        central.append(np.linalg.norm(g - gc) / np.linalg.norm(g))
        onesided.append(np.linalg.norm(g - go) / np.linalg.norm(g))
    return np.array(central), np.array(onesided)


def starts(n=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = THETA_SOD.copy()
        for k in (0, 1, 3, 4):
            t[k] *= np.exp(0.4 * rng.standard_normal())
        t[2] += 0.2 * rng.standard_normal()
        t[5] += 0.2 * rng.standard_normal()
        out.append(t)
    return np.array(out)


def invert(value_grad, theta0, maxiter=300):
    """L-BFGS-B on (L, dL/dtheta); rho and P bounded below. History per gradient evaluation."""
    hist = []

    def f(t):
        L, g = value_grad(t)
        hist.append((L, *t))
        return L, g

    bounds = [(1e-3, None), (1e-3, None), (None, None), (1e-3, None), (1e-3, None), (None, None)]
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-16, gtol=1e-14))
    return res.x, np.array(hist), int(res.nit)


def main(argv):
    binary, out = argv[1], argv[2]
    flags = [a for a in argv[3:] if a.startswith("--")]
    tube = Tube(binary, *flags)
    r = {}
    # 1. the gradient, both modes
    J, g_adj, ms_fwd_tape, ms_adj = tube.adjoint("J", THETA_SOD)
    _, g_tan, ms_tan = tube.tangent("J", THETA_SOD)
    t0 = time.perf_counter()
    for _ in range(5):
        tube.J(THETA_SOD)
    ms_fwd = (time.perf_counter() - t0) / 5 * 1e3  # includes the pipe; the binary's own times below are the ones to quote
    reps = [tube.adjoint("J", THETA_SOD) for _ in range(5)]
    ms_fwd_tape = np.median([x[2] for x in reps])
    ms_adj = np.median([x[3] for x in reps])
    ms_tan = np.median([tube.tangent("J", THETA_SOD)[2] for _ in range(5)])
    r.update(J=J, g_adjoint=g_adj, g_tangent=g_tan, ms_forward_pipe=ms_fwd, ms_forward_tape=ms_fwd_tape, ms_adjoint=ms_adj,
             ms_six_tangents=ms_tan)
    print(f"J = {J:.15e}")
    for k in range(6):
        print(f"  dJ/d{NAMES[k]:6s} adjoint {g_adj[k]: .15e}  tangent {g_tan[k]: .15e}")
    print(f"adjoint against tangent: {np.max(np.abs(g_adj - g_tan)) / np.max(np.abs(g_adj)):.2e}")
    print(f"ms: forward with tape {ms_fwd_tape:.2f}, adjoint {ms_adj:.2f}, six tangent runs {ms_tan:.2f}")
    # 2. the FD sweep
    central, onesided = fd_sweep(lambda t: tube.J(t)[0], THETA_SOD, g_adj)
    r.update(hs=np.array(HS), fd_central=central, fd_onesided=onesided)
    for h, c, o in zip(HS, central, onesided):
        print(f"  h {h:.0e}: central {c:.2e}, closer one-sided {o:.2e}")
    # 3. the sensitivity map and its FD check at a few cells
    path = out.replace(".npz", "_map.bin")
    tube.ask(f"dump {path} J " + tube.th(THETA_SOD))
    rows = np.fromfile(path, dtype=np.float64).reshape(-1, 7)
    r.update(x=rows[:, 0], rho=rows[:, 1], u=rows[:, 2], P=rows[:, 3], dJdU0=rows[:, 4:7])
    n = rows.shape[0]
    dx = 1.0 / n
    checks = []
    for i in (20, 60, 99, 100, 140, 180):
        for v in range(3):
            h = 1e-5
            Jp = float(tube.ask(f"fcell {i} {v} {h} " + tube.th(THETA_SOD))[1])
            Jm = float(tube.ask(f"fcell {i} {v} {-h} " + tube.th(THETA_SOD))[1])
            fd = (Jp - Jm) / (2 * h) / dx  # per unit length, as the map
            checks.append((i, v, rows[i, 4 + v], fd))
    checks = np.array(checks)
    r["map_checks"] = checks
    err = np.abs(checks[:, 2] - checks[:, 3]) / np.max(np.abs(rows[:, 4:7]))
    print(f"map against central differences at {len(checks)} entries: largest {err.max():.2e} of the map's largest")
    # 4. the inverse problem
    tube.ask("target " + tube.th(THETA_SOD))

    def vg(t):
        L, g, _, _ = tube.adjoint("L", t)
        return L, g

    S = starts()
    finals, hists, nits = [], [], []
    t0 = time.perf_counter()
    for s in S:
        th, hist, nit = invert(vg, s)
        finals.append(th)
        hists.append(hist)
        nits.append(nit)
        print(f"  start {np.array2string(s, precision=3)} -> {np.array2string(th, precision=6)}, L {hist[-1, 0]:.2e}, "
              f"|theta - true| {np.max(np.abs(th - THETA_SOD)):.2e}, {len(hist)} gradients")
    wall = time.perf_counter() - t0
    r.update(starts=S, finals=np.array(finals), nits=np.array(nits), inversion_wall_s=wall,
             gradients=np.array([len(h) for h in hists]))
    L = max(len(h) for h in hists)
    H = np.full((len(hists), L, 7), np.nan)
    for k, h in enumerate(hists):
        H[k, :len(h)] = h
    r["histories"] = H
    tube.close()
    np.savez(out, **r)
    print(f"wrote {out}; inversion wall {wall:.1f} s for {sum(len(h) for h in hists)} gradients")


if __name__ == "__main__":
    main(sys.argv)
