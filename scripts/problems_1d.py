"""The 1D problems for Python: the table of src/hydro/problems_1d.hpp, LUNA's initial state of each, the exact solution.

Run with LUNA's environment (luna.hydro.exact_riemann). Sources in LUNA, tests/: test_hydro_accuracy_shocktube.py (TORO, run_1d),
test_hydro_hard_einfeldt.py, test_hydro_hard_la14379.py (CASES, VACUUM), test_hydro_hard_leblanc.py, test_hydro_hard_noh_planar.py
(initial, exact, SKIP, X_MAX), test_hydro_hard_titarev_toro.py, test_hydro_accuracy_linear_waves.py (initial),
test_hydro_accuracy_smooth.py (sound_advected), src/luna/ic_generation/. Not in LUNA: simple-wave, shock-<Mach>.

exact(name, a, b, eps) gives (rho, u, p) of the cells [a[i], b[i]] at t_end, conservative averages, by LUNA's rule where it has one:
256 midpoints a cell (exact_riemann.cell_average). Shu-Osher, Woodward-Colella and Titarev-Toro have a stored reference instead.
"""
import numpy as np
from luna.hydro import exact_riemann as er

OUTFLOW, REFLECTIVE, PERIODIC = 1, 2, 0
G14, G53 = 1.4, 5.0 / 3.0
EPS = 1e-6


def tube(WL, WR, x0, box, t, gamma=G14):
    return dict(kind="tube", WL=WL, WR=WR, x0=x0, box=box, t=t, gamma=gamma, boundary=OUTFLOW)


PROBLEMS = {f"toro-{k}": tube(WL, WR, x0, (0.0, 1.0), t) for k, (WL, WR, x0, t) in er.TORO.items()}
PROBLEMS.update({
    "einfeldt-strong": tube((1.0, -2.0, 0.4 * (3.0 - 0.5 * 4.0)), (1.0, 2.0, 0.4 * (3.0 - 0.5 * 4.0)), 0.5, (0.0, 1.0), 0.15),
    "einfeldt-shear": tube((1.0, -1.0, 0.4 * (5.0 - 0.5 * 1.0)), (1.0, 1.0, 0.4 * (5.0 - 0.5 * 1.0)), 0.5, (0.0, 1.0), 0.15),
    "la-peak": tube((0.1261192, 8.9047029, 782.92899), (6.591493, 2.2654207, 3.1544874), 0.5, (0.1, 0.6), 3.9e-3),
    "la-rcvcr": tube((1.0, -4.0, 0.4), (1.0, 4.0, 0.4), 0.5, (0.0, 1.0), 0.15),
    "la-stationary": tube((1.0, -19.59745, 1e3), (1.0, -19.59745, 1e-2), 0.8, (0.0, 1.0), 0.012),
    "la-slow": tube((3.857143, -0.810631, 10.33333), (1.0, -3.44, 1.0), 0.5, (0.0, 1.0), 1.0),
    "la-streams": tube((1.0, 2.0, 0.1), (1.0, -2.0, 0.1), 0.5, (0.0, 1.0), 0.8),
    "la-vacuum": tube((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), 0.0, (-5.0, 1.0), 0.75),
    "la-near-vacuum": tube((1e-6, 0.0, 1e-6), (1.0, 0.0, 1.0), 0.0, (-5.0, 1.0), 0.75),
    "leblanc": tube((1.0, 0.0, (2.0 / 3.0) * 1e-1), (1e-3, 0.0, (2.0 / 3.0) * 1e-10), 0.3, (0.0, 1.0), 0.5, G53),
    "noh": dict(kind="noh", box=(0.0, 2.0), t=0.6, gamma=G53, boundary=REFLECTIVE),
    "shu-osher": dict(kind="reference", luna="Shu-Osher", box=(-5.0, 5.0), t=1.8, gamma=G14, boundary=OUTFLOW),
    "titarev-toro": dict(kind="reference", luna="Titarev-Toro", box=(-5.0, 5.0), t=5.0, gamma=G14, boundary=OUTFLOW),
    "woodward-colella": dict(kind="reference", luna="Woodward-Colella", box=(0.0, 1.0), t=0.038, gamma=G14, boundary=REFLECTIVE),
    "sound-right": dict(kind="eigen", mode=(0.0, 1.0, 1.0, 1.0), box=(0.0, 1.0), t=1.0, gamma=G53, boundary=PERIODIC),
    "sound-left": dict(kind="eigen", mode=(0.0, 1.0, -1.0, 1.0), box=(0.0, 1.0), t=1.0, gamma=G53, boundary=PERIODIC),
    "entropy": dict(kind="eigen", mode=(0.5, 1.0, 0.0, 0.0), box=(0.0, 1.0), t=2.0, gamma=G53, boundary=PERIODIC),
    "sound-advected": dict(kind="advected", box=(0.0, 1.0), t=1.0, gamma=G53, boundary=PERIODIC),
    "simple-wave": dict(kind="simple", amp=0.1, box=(0.0, 1.0), t=0.2, gamma=G14, boundary=PERIODIC),
})
MACHS = ("1.2", "1.6556", "2", "3", "5", "10")


def problem(name):
    if name == "sod":
        name = "toro-1"
    if name.startswith("shock-"):
        g, m = G14, float(name[6:])
        s = m * np.sqrt(g)
        rho2, p2 = (g + 1) * m * m / ((g - 1) * m * m + 2), (2 * g * m * m - (g - 1)) / (g + 1)
        return dict(tube((rho2, s * (1 - 1 / rho2), p2), (1.0, 0.0, 1.0), 0.1, (0.0, 1.0), 0.5 / s), kind="shock", speed=s)
    return PROBLEMS[name]


LINEAR = ("sound-right", "sound-left", "entropy", "sound-advected")


def conserved(rho, u, p, g):
    return np.stack([rho, rho * u, p / (g - 1.0) + 0.5 * rho * u * u])


def primitive(U, g):
    """(rho, u, p), u = 0 in empty cells."""
    rho, mom, E = U
    u = np.divide(mom, rho, out=np.zeros_like(rho), where=rho > 0)
    return rho, u, (g - 1.0) * (E - 0.5 * rho * u * u)


def eigenmode(pb, a, b, eps):
    """The conserved cell averages of test_hydro_accuracy_linear_waves.py, initial(), on any cells."""
    v0, drho, dv, dp = pb["mode"]
    g, k, h = pb["gamma"], 2.0 * np.pi, b - a
    half = 0.5 * k * h
    s = np.sin(k * 0.5 * (a + b)) * np.sin(half) / half
    return np.stack([1.0 + eps * drho * s, v0 + eps * (dv + v0 * drho) * s,
                     (1.0 / g) / (g - 1.0) + 0.5 * v0 * v0 + eps * (dp / (g - 1.0) + 0.5 * v0 * v0 * drho + v0 * dv) * s])


def initial(name, n, eps=EPS):
    """The conserved state of LUNA's run on n uniform cells, (3, n), from its own formulas; None: LUNA's registered IC."""
    pb = problem(name)
    lo, hi = pb["box"]
    edges = np.linspace(lo, hi, n + 1)
    a, b, g = edges[:-1], edges[1:], pb["gamma"]
    if pb["kind"] in ("tube", "shock"):
        f = np.clip((pb["x0"] - a) / (b - a), 0.0, 1.0)
        return f[None, :] * conserved(*[np.float64(v) for v in pb["WL"]], g)[:, None] \
            + (1.0 - f[None, :]) * conserved(*[np.float64(v) for v in pb["WR"]], g)[:, None]
    x = 0.5 * (a + b)
    if pb["kind"] == "noh":
        u = -1.0 * np.clip((2.0 - x) / (2.0 - 1.0), 0.0, 1.0)
        return np.stack([np.ones_like(x), u, 1e-6 / (g - 1.0) + 0.5 * u * u])
    if pb["kind"] == "eigen":
        return eigenmode(pb, a, b, eps)
    if pb["kind"] == "advected":
        s = eps * np.sin(2.0 * np.pi * x)
        return conserved(1.0 + s, 1.0 + s, 1.0 + s, g)
    if pb["kind"] == "simple":
        X, W = np.polynomial.legendre.leggauss(4)
        return sum(0.5 * w * conserved(*simple_exact(pb, x + 0.5 * (b - a) * q, 0.0), g) for q, w in zip(X, W))
    return None


def vacuum_sample(WL, WR, g, xi):
    """(rho, u, p) of a Riemann problem that generates vacuum (Toro sec. 4.6); test_hydro_hard_la14379.py, _vacuum_sample."""
    (_, ul, _), (_, ur, _) = WL, WR
    al, ar = (np.sqrt(g * W[2] / W[0]) for W in (WL, WR))
    sl, sr = ul + 2.0 * al / (g - 1.0), ur - 2.0 * ar / (g - 1.0)
    out = [np.zeros_like(xi) for _ in range(3)]
    for W, s, region, fan in ((WL, -1.0, xi < ul - al, (xi >= ul - al) & (xi < sl)),
                              (WR, 1.0, xi >= ur + ar, (xi > sr) & (xi < ur + ar))):
        f = er._fan(W, xi[fan], g, s)
        for k in range(3):
            out[k][region], out[k][fan] = W[k], f[k]
    return tuple(out)


def vacuum_left(WR, g, xi):
    """The right gas expanding into a true vacuum on the left (Toro sec. 4.6)."""
    rho, u, p = WR
    a = np.sqrt(g * p / rho)
    out = [np.zeros_like(xi) for _ in range(3)]
    right, fan = xi >= u + a, (xi > u - 2.0 * a / (g - 1.0)) & (xi < u + a)
    f = er._fan(WR, xi[fan], g, 1.0)
    for k in range(3):
        out[k][right], out[k][fan] = WR[k], f[k]
    return tuple(out)


def sample(pb, xi):
    WL, WR, g = pb["WL"], pb["WR"], pb["gamma"]
    if WL[0] == 0.0:
        return vacuum_left(WR, g, xi)
    if 2.0 * sum(np.sqrt(g * W[2] / W[0]) for W in (WL, WR)) / (g - 1.0) < WR[1] - WL[1]:
        return vacuum_sample(WL, WR, g, xi)
    return er.sample(WL, WR, g, xi)


def averages(pb, a, b, sub=256):
    """Conservative averages of the Riemann solution on the cells, 256 midpoints each (exact_riemann.cell_average)."""
    g, x0, t = pb["gamma"], pb["x0"], pb["t"]
    f = (np.arange(sub) + 0.5) / sub
    out = np.empty((3, a.size))
    for s in range(0, a.size, 4096):
        A, B = a[s:s + 4096], b[s:s + 4096]
        x = A[:, None] + (B - A)[:, None] * f[None, :]
        rho, u, p = sample(pb, (x - x0) / t)
        out[0, s:s + 4096], out[1, s:s + 4096] = rho.mean(1), (rho * u).mean(1)
        out[2, s:s + 4096] = (p / (g - 1.0) + 0.5 * rho * u * u).mean(1)
    return primitive(out, g)


def simple_exact(pb, x, t):
    """The isentropic simple wave at (x, t), along its characteristics: (rho, u, p)."""
    amp, g = pb["amp"], pb["gamma"]
    x0 = (x - t) % 1.0
    for _ in range(60 if t > 0 else 1):
        c = 1.0 + amp * np.sin(2.0 * np.pi * x0)
        f = x0 + (6.0 * c - 5.0) * t - x
        f = f - np.round(f)
        x0 = x0 - f / (1.0 + 6.0 * t * amp * 2.0 * np.pi * np.cos(2.0 * np.pi * x0))
    c = 1.0 + amp * np.sin(2.0 * np.pi * x if t == 0 else 2.0 * np.pi * x0)
    rho = c ** 5
    return rho, 5.0 * (c - 1.0), rho ** g / g


def noh_exact(pb, a, b):
    """test_hydro_hard_noh_planar.py, exact(), on any cells: the shocked fraction of each."""
    g, t = pb["gamma"], pb["t"]
    D, rho_s = (g - 1.0) / 2.0, (g + 1.0) / (g - 1.0)
    p_s = (g - 1.0) / 2.0 * rho_s
    f = np.clip((D * t - a) / (b - a), 0.0, 1.0)
    return f * rho_s + (1 - f) * 1.0, (1 - f) * -1.0, f * p_s + (1 - f) * 1e-6


def exact(name, a, b, eps=EPS):
    """(rho, u, p) of the cells [a, b] at t_end."""
    pb = problem(name)
    kind, g = pb["kind"], pb["gamma"]
    if kind in ("tube", "shock"):
        return averages(pb, a, b)
    if kind == "noh":
        return noh_exact(pb, a, b)
    if kind == "eigen":
        return primitive(eigenmode(pb, a, b, eps), g)     # one period later
    if kind == "advected":
        x, t, k, c = 0.5 * (a + b), pb["t"], 2.0 * np.pi, np.sqrt(g)
        s0, sp, sm = (eps * np.sin(k * (x - v * t)) for v in (1.0, 1.0 + c, 1.0 - c))
        ap, am = 0.5 * (1.0 + c) / c ** 2, 0.5 * (1.0 - c) / c ** 2
        return 1.0 + (1.0 - 1.0 / c ** 2) * s0 + ap * sp + am * sm, 1.0 + c * (ap * sp - am * sm), 1.0 + c * c * (ap * sp + am * sm)
    if kind == "simple":
        X, W = np.polynomial.legendre.leggauss(4)
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        return primitive(sum(0.5 * w * conserved(*simple_exact(pb, mid + half * q, pb["t"]), g) for q, w in zip(X, W)), g)
    raise KeyError(name)
