"""A word out of colliding blast waves: LUNA's blast-emergence example (LUNA scripts/examples/autodiff/luna_emergence.py, the
blast driver) through luna-cpp's hand-written adjoint (rollout binary), static or moving mesh.

The setup is LUNA's: 256^2 periodic unit box, gamma = 5/3, a uniform medium rho = 1 at rest; the free parameter is the internal
energy per volume of every cell, e(x, 0) = 1 + 49 sigmoid(theta) (65 536 parameters), theta0 = -3 + 0.3 N(0, 1) (numpy
RandomState(0), as LUNA's); 300 fixed steps of dt = 0.25 dx / c_s(e = 50); loss mean over cells of (e(x, T) - target)^2, the
target the word "LUNA" rendered to [1, 4]; Adam, lr 0.2, 200 iterations. What cannot match: luna-cpp's 2D limiter is the
face-centroid one (LUNA: MC); on the moving mesh the cells move, so the loss compares each cell's e with the target at its
generator (bilinear, periodic) and the gradient includes the positions and the volumes.
    python blast_emergence.py <rollout binary> <out.npz> <workdir> [--moving] [--n=256] [--steps=300] [--iters=200] [--word=LUNA]
"""
import sys
import time

import numpy as np

from rollout_client import Rollout

GAMMA, E_FLOOR, E_MAX = 5.0 / 3.0, 1.0, 50.0


def render(n, text, lo, hi):
    """The word in white on black, 400 px square, resized to n (linear), mapped to [lo, hi]; [i_x, i_y] order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color="white", fontsize=260.0 / max(1, len(text)), fontweight="bold",
            transform=ax.transAxes)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float64) / 255.0
    plt.close(fig)
    r = np.clip(zoom(img, n / img.shape[0], order=1), 0, 1)
    return lo + (hi - lo) * r[::-1].T


def bilinear(field, x):
    """Periodic bilinear interpolation of field (n, n) at points x (N, 2) in [0, 1)^2, and its gradient."""
    n = field.shape[0]
    g = x * n - 0.5
    i0 = np.floor(g).astype(int)
    f = g - i0
    i0 %= n
    i1 = (i0 + 1) % n
    a, b = field[i0[:, 0], i0[:, 1]], field[i1[:, 0], i0[:, 1]]
    c, d = field[i0[:, 0], i1[:, 1]], field[i1[:, 0], i1[:, 1]]
    fx, fy = f[:, 0], f[:, 1]
    val = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
    gx = n * ((b - a) * (1 - fy) + (d - c) * fy)
    gy = n * ((c - a) * (1 - fx) + (d - b) * fx)
    return val, np.stack([gx, gy], axis=1)


def main(argv):
    binary, out, work = argv[1:4]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv[4:])
    moving, n, steps, iters = "moving" in opts, int(opts.get("n", 256)), int(opts.get("steps", 300)), int(opts.get("iters", 200))
    word, lr = opts.get("word", "LUNA"), float(opts.get("lr", 0.2))
    dx = 1.0 / n
    dt = 0.25 * dx / np.sqrt(GAMMA * (GAMMA - 1.0) * E_MAX)
    tgt = render(n, word, E_FLOOR, 4.0 * E_FLOOR)
    N = n * n
    r = Rollout(binary, 2, n, steps, dt, work, moving=moving, tag="blast")

    def U0_of(theta):
        U = np.zeros((n, n, 4))
        U[..., 0] = 1.0
        U[..., 3] = E_FLOOR + (E_MAX - E_FLOOR) / (1.0 + np.exp(-theta))
        return U

    def value_grad(theta):
        f = r.forward(U0_of(theta))
        Q, vol = f["Q"], f["vol"]
        ke = 0.5 * (Q[:, 1] ** 2 + Q[:, 2] ** 2) / Q[:, 0]
        e = (Q[:, 3] - ke) / vol
        if moving:
            t, tg = bilinear(tgt, f["x"])
        else:
            t, tg = tgt.reshape(N), None
        d = e - t
        L = np.mean(d ** 2)
        ebar = 2.0 * d / N
        Qbar = np.zeros_like(Q)
        Qbar[:, 3] = ebar / vol
        Qbar[:, 1] = -ebar / vol * Q[:, 1] / Q[:, 0]
        Qbar[:, 2] = -ebar / vol * Q[:, 2] / Q[:, 0]
        Qbar[:, 0] = ebar / vol * ke / Q[:, 0]
        xbar = -ebar[:, None] * tg if moving else None
        volbar = -ebar * e / vol if moving else None
        U0bar, ms = r.adjoint(Qbar, xbar, volbar)
        s = 1.0 / (1.0 + np.exp(-theta))
        return L, U0bar[..., 3] * (E_MAX - E_FLOOR) * s * (1 - s), f["ms"], ms, f["violations"], e

    theta = -3.0 + 0.3 * np.random.RandomState(0).standard_normal((n, n))
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    losses, t_f, t_a, viol = [], [], [], 0
    # the gradient against central differences at a few cells, at the start
    L0, g0, *_ = value_grad(theta)
    checks = []
    for (i, j) in [(n // 2, n // 2), (n // 4, n // 3), (10, 200 % n)]:
        h = 1e-5
        tp, tm = theta.copy(), theta.copy()
        tp[i, j] += h
        tm[i, j] -= h
        checks.append((i, j, g0[i, j], (value_grad(tp)[0] - value_grad(tm)[0]) / (2 * h)))
    checks = np.array(checks)
    print(f"{'moving' if moving else 'static'} {n}^2, {steps} steps of dt {dt:.4e}; L0 {L0:.4e}; gradient against central "
          f"differences at 3 cells: {np.array2string(np.abs(checks[:, 2] - checks[:, 3]) / np.abs(checks[:, 3]), precision=2)} relative")
    t0 = time.perf_counter()
    for it in range(iters):
        L, g, ms_f, ms_a, vi, e = value_grad(theta)
        losses.append(L)
        t_f.append(ms_f)
        t_a.append(ms_a)
        viol += vi
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta = theta - lr * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
        if it % 20 == 0 or it == iters - 1:
            print(f"  it {it:4d}  L = {L:.4e}  (forward {ms_f:.0f} ms, adjoint {ms_a:.0f} ms)", flush=True)
    wall = time.perf_counter() - t0
    f = r.forward(U0_of(theta))
    Q = f["Q"]
    e_final = (Q[:, 3] - 0.5 * (Q[:, 1] ** 2 + Q[:, 2] ** 2) / Q[:, 0]) / f["vol"]
    peak, _ = r.memory()
    r.close()
    np.savez(out, losses=np.array(losses), theta=theta, e0=U0_of(theta)[..., 3], e_final=e_final, x_final=f["x"], target=tgt,
             ms_forward=np.array(t_f), ms_adjoint=np.array(t_a), wall_s=wall, peak_bytes=peak, cfl_violations=viol, checks=checks,
             n=n, steps=steps, dt=dt, moving=moving)
    print(f"wrote {out}: L {losses[0]:.4e} -> {losses[-1]:.4e} in {iters} iterations, {wall:.0f} s "
          f"({wall / iters * 1e3:.0f} ms a gradient with the pipe; forward {np.median(t_f):.0f} ms, adjoint {np.median(t_a):.0f} ms); "
          f"device peak {peak / 2**20:.0f} MiB ({peak / N:.0f} B/cell); CFL violations {viol}")


if __name__ == "__main__":
    main(sys.argv)
