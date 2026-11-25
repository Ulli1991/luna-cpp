"""The rollout binary (src/examples/rollout.cpp) in serve mode, for the optimisers of the 2D and 3D examples.

Cells are the n^D lattice of the periodic unit box in C order (the last axis fastest): an array of shape (n,) * D + (V,) indexed
[i_x, i_y(, i_z)] is the state of the cell at ((i_x + 1/2) / n, ...). V = D + 2: rho, momentum (D), total energy per volume.
"""
import os
import subprocess

import numpy as np


class Rollout:
    def __init__(self, binary, dim, n, steps, dt, workdir, moving=False, gamma=5.0 / 3.0, courant=0.3, tag="r"):
        self.dim, self.n, self.steps, self.dt, self.moving = dim, n, steps, dt, moving
        self.V = dim + 2
        self.N = n ** dim
        os.makedirs(workdir, exist_ok=True)
        self.f = {k: os.path.join(workdir, f"{tag}_{k}.bin") for k in ("u0", "du0", "fin", "bars", "out")}
        flags = [f"--dim={dim}", f"--n={n}", f"--steps={steps}", f"--dt={dt!r}", f"--gamma={gamma!r}", f"--courant={courant!r}"]
        if moving:
            flags.append("--moving")
        self.p = subprocess.Popen([binary, *flags], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def ask(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().split()
        if not out or out[0] != "ok":
            raise RuntimeError(f"rollout: {line} -> {out}")
        return out[1:]

    def _write(self, key, a):
        np.ascontiguousarray(a, dtype=np.float64).tofile(self.f[key])

    def forward(self, U0):
        """U0: (n,)*D + (V,), intensive. Returns Q (extensive), x, vol at the end (cell order), ms, CFL violations."""
        self._write("u0", U0.reshape(self.N, self.V))
        ms, viol = self.ask(f"forward {self.f['u0']} {self.f['fin']}")
        rows = np.fromfile(self.f["fin"]).reshape(self.N, self.V + self.dim + 1)
        return dict(Q=rows[:, :self.V], x=rows[:, self.V:self.V + self.dim], vol=rows[:, -1], ms=float(ms), violations=int(viol))

    def adjoint(self, Qbar, xbar=None, volbar=None):
        """Bars of the last forward's end -> dL/dU0, shape (n,)*D + (V,); ms."""
        bars = np.zeros((self.N, self.V + self.dim + 1))
        bars[:, :self.V] = Qbar
        if xbar is not None:
            bars[:, self.V:self.V + self.dim] = xbar
        if volbar is not None:
            bars[:, -1] = volbar
        self._write("bars", bars)
        (ms,) = self.ask(f"adjoint {self.f['bars']} {self.f['out']}")
        return np.fromfile(self.f["out"]).reshape((self.n,) * self.dim + (self.V,)), float(ms)

    def tangent(self, U0, dU0):
        """dQ at the end (extensive, cell order) along dU0 (intensive); ms."""
        self._write("u0", U0.reshape(self.N, self.V))
        self._write("du0", dU0.reshape(self.N, self.V))
        (ms,) = self.ask(f"tangent {self.f['u0']} {self.f['du0']} {self.f['out']}")
        return np.fromfile(self.f["out"]).reshape(self.N, self.V), float(ms)

    def cfl(self, U0):
        self._write("u0", U0.reshape(self.N, self.V))
        return float(self.ask(f"cfl {self.f['u0']}")[0])

    def memory(self):
        peak, now = self.ask("memory")
        return int(peak), int(now)

    def close(self):
        self.p.stdin.write("quit\n")
        self.p.stdin.flush()
        self.p.wait()


def centres(n, dim):
    """Cell-centre coordinate arrays, each (n,)*dim, 'ij' indexing."""
    c = (np.arange(n) + 0.5) / n
    return np.meshgrid(*([c] * dim), indexing="ij")
