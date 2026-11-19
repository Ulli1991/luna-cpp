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
# The super-hard tube (research_notes/super_tube_2026-11-10.md), the table of problems_1d.hpp: from x on, (rho, v, P), up to the next x.
SUPER_SEGMENTS = ((0.0, (1, 0, 1)), (6.0, (10, 0, 1)), (6.05, (1, 0, 1)), (8.05, (1, 1, 1)), (8.80, (10, 1, 1)), (8.85, (1, 1, 1)),
                  (9.60, (1, 0, 1)), (11.60, (1, 0.75, 1)), (12.10, (0.125, 0, 0.1)), (12.60, (1, 0, 1)), (14.60, (1, 0, 0.1)),
                  (15.10, (1e-3, 0, 1e-10)), (15.60, (1, 0, 1)), (17.60, (3.857143, -0.810631, 10.33333)), (18.10, (1, -3.44, 1)),
                  (18.60, (1, 0, 1)), (22.60, (5.99924, 19.5975, 460.894)), (22.90, (5.99242, -6.19633, 46.0950)), (23.20, (1, 0, 1)),
                  (27.20, (1, -2, 0.4)), (27.70, (1, 2, 0.4)), (28.20, (1, 0, 1)), (32.20, (1, 0, 1000)), (32.30, (1, 0, 0.01)),
                  (33.10, (1, 0, 100)))
SUPER_LENGTH, SUPER_WAVE, SUPER_WAVELENGTH = 33.2, 1e-4, 0.25
# The stages, (name, start, end): A to G single Riemann problems to t = 0.129 (scripts/super_tube_layout.py).
SUPER_STAGES = (("A", 0.0, 1.0), ("B", 3.0, 4.0), ("C", 6.0, 6.05), ("C2", 8.05, 9.60), ("D", 11.60, 12.60), ("E", 14.60, 15.60),
                ("F", 17.60, 18.60), ("S", 22.60, 23.20), ("G", 27.20, 28.20), ("H", 32.20, 33.20))
PROBLEMS["super-tube"] = dict(kind="super", box=(0.0, SUPER_LENGTH), t=2.0, gamma=G14, boundary=OUTFLOW, right=REFLECTIVE)
MACHS = ("1.2", "1.6556", "2", "3", "5", "10")
# Stage L of the tube v2 alone (problems_1d.hpp, research_notes/stage_l_2026-11-19.md): (1, 0, 0.1) | (r, 0, 1e-7 r) at 0 on [-2, 9],
# t = 4, exact by the bisection solver (riemann(), pressures down to 1e-16).
STAGE_L = {"3": (1e-3, 1e-10), "6": (1e-6, 1e-13), "9": (1e-9, 1e-16)}
for _k, (_r, _p) in STAGE_L.items():
    PROBLEMS[f"stage-l-{_k}"] = dict(tube((1.0, 0.0, 0.1), (_r, 0.0, _p), 0.0, (-2.0, 9.0), 4.0), bisection=True)


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
    if pb["kind"] == "super":
        return super_initial(a, b)
    if pb["kind"] == "super2":
        return super2_initial(a, b, pb.get("m0", False))
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


def super_wave(a, b, t):
    """Stage B at t, per unit A: the integral over [a, b] of the entropy wave at rest and of the sound wave carried at c, over b - a."""
    k, c = 2.0 * np.pi / SUPER_WAVELENGTH, np.sqrt(G14)

    def part(shift):
        lo, hi = np.maximum(a, 3.0 + shift), np.minimum(b, 4.0 + shift)
        return np.where(hi > lo, (np.cos(k * (lo - 3.0 - shift)) - np.cos(k * (hi - 3.0 - shift))) / k, 0.0) / (b - a)
    return part(0.0), part(c * t)


def super_initial(a, b):
    """The conserved cell averages of problems_1d.hpp, super_tube(): volume fractions; stage B linear in U."""
    ends = [x for x, _ in SUPER_SEGMENTS[1:]] + [SUPER_LENGTH]
    U = sum(np.clip(np.minimum(b, e) - np.maximum(a, x), 0.0, None) / (b - a) * conserved(*map(np.float64, W), G14)[:, None]
            for (x, W), e in zip(SUPER_SEGMENTS, ends))
    s, _ = super_wave(a, b, 0.0)
    c = np.sqrt(G14)
    return U + SUPER_WAVE * s * np.array([2.0, c, G14 / (G14 - 1.0)])[:, None]


def super_fans(t):
    """(x0, WL, WR, lo, hi) of every interface of the segments, [lo, hi] its waves at t (a contact alone: lo = hi)."""
    out = []
    for (_, WL), (x0, WR) in zip(SUPER_SEGMENTS, SUPER_SEGMENTS[1:]):
        w = er.waves(tuple(map(float, WL)), tuple(map(float, WR)), G14)
        left = min(w["left"]) if abs(w["p"] / WL[2] - 1) > 1e-12 else w["u"]   # no wave on a side whose pressure stays
        right = max(w["right"]) if abs(w["p"] / WR[2] - 1) > 1e-12 else w["u"]
        out.append((x0, WL, WR, x0 + left * t, x0 + right * t))
    return out


def super_meets(t):
    """Pairs of neighbouring interfaces whose waves overlap at t, by x0: [] while each stays a single Riemann problem."""
    fans = super_fans(t)
    return [(A[0], B[0]) for A, B in zip(fans, fans[1:]) if A[4] > B[3]]


SUPER_EXACT = 22.6   # A to F; G's interfaces too. S and H interact within themselves at once: not exact
SUPER_G = (27.2, 27.7, 28.2)


def super_exact(a, b, t, sub=256):
    """(rho, u, p) averages of the cells at t < 0.129 on A to G: each interface's Riemann solution inside its waves, between them
    the segment's state, stage B linear. Cells of S and H: not exact (each interface's Riemann solution alone)."""
    f = (np.arange(sub) + 0.5) / sub
    out = np.empty((3, a.size))
    fans = super_fans(t)
    his = np.array([hi for *_, hi in fans])
    states = [np.array([w[i] for _, w in SUPER_SEGMENTS], float) for i in range(3)]
    for s in range(0, a.size, 4096):
        A, B = a[s:s + 4096], b[s:s + 4096]
        x = A[:, None] + (B - A)[:, None] * f[None, :]
        k = np.searchsorted(np.sort(his), x, side="right")     # the segment between two interfaces' waves
        W = [st[k] for st in states]
        for x0, WL, WR, lo, hi in fans:
            m = (x >= lo) & (x < hi)
            if m.any() and (x0 < SUPER_EXACT or x0 in SUPER_G):
                fan = er.sample(tuple(map(float, WL)), tuple(map(float, WR)), G14, (x[m] - x0) / t)
                for i in range(3):
                    W[i][m] = fan[i]
        out[:, s:s + 4096] = conserved(*W, G14).mean(axis=2)
    e, sound = super_wave(a, b, t)
    c = np.sqrt(G14)
    out += SUPER_WAVE * (e * np.array([1.0, 0.0, 0.0])[:, None] + sound * np.array([1.0, c, G14 / (G14 - 1.0)])[:, None])
    return primitive(out, G14)


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
    if pb.get("bisection"):
        return riemann(WL, WR, xi, g)
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
    if kind == "super":
        return super_exact(a, b, 0.1)
    if kind == "super2":
        return super2_exact(a, b, SUPER2_T1)
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


# ---- The super-hard tube v2 (research_notes/super_tube_v2_2026-11-18.md) -------------------------------------------------------
# scripts/super_tube_layout.py v2() is the one source of the layout here; src/hydro/problems_1d.hpp holds the same numbers.
import super_tube_layout as lay  # noqa: E402

_V2 = lay.v2()
SUPER2_SEGMENTS = tuple((x, s) for x, _, s in _V2["segments"])
SUPER2_LENGTH = _V2["length"]
SUPER2_STAGES = tuple((k, *_V2["stages"][k]) for k, *_ in lay.STAGES_V2)
SUPER2_B = _V2["stages"]["B"][0]                         # stage B's waves on [B, B + 1], v1's
SUPER2_SO = (_V2["stages"]["SO"][0] + 1.0, _V2["stages"]["SO"][0] + 1.0 + 2.0 * np.pi / lay.SO_K * lay.SO_PERIODS)  # shock at x_s, sine to end
SUPER2_T1, SUPER2_T_END = lay.T1, 4.0
SUPER2_H4096 = SUPER2_LENGTH / 4096
SUPER2_SINGLE = ("M", "L", "SO", "N", "V")               # the new stages, each also alone (super-v2-<key>)
PROBLEMS["super-tube-v2"] = dict(kind="super2", box=(0.0, SUPER2_LENGTH), t=SUPER2_T_END, gamma=G14, boundary=OUTFLOW, right=REFLECTIVE)
# A diagnostic: stage M's stream and bump replaced by the ambient gas (the moving mesh stops at M at once, luna-cpp #14).
PROBLEMS["super-tube-v2-m0"] = dict(PROBLEMS["super-tube-v2"], m0=True)


def super2_region(key):
    """(lo, hi) where a stage of v2 is scored: between the middles of the quiet lines either side at t = 0.1 (super_tube_layout.v2)."""
    return _V2["regions"][key]


def super2_window(key):
    """A stage of v2 alone: from the middle of the buffer on its left to the middle of the one on its right, on the cells of the tube
    at N = 4096, (first cell, cells); problems_1d.hpp has the result."""
    i = [k for k, *_ in SUPER2_STAGES].index(key)
    lo = 0.0 if i == 0 else 0.5 * (SUPER2_STAGES[i - 1][2] + SUPER2_STAGES[i][1])
    hi = SUPER2_LENGTH if i + 1 == len(SUPER2_STAGES) else 0.5 * (SUPER2_STAGES[i][2] + SUPER2_STAGES[i + 1][1])
    first = int(np.floor(lo / SUPER2_H4096 + 1e-9))
    return first, int(np.ceil(hi / SUPER2_H4096 - 1e-9)) - first


for _k in SUPER2_SINGLE:
    _first, _cells = super2_window(_k)
    PROBLEMS[f"super-v2-{_k.lower()}"] = dict(kind="super2", box=(_first * SUPER2_H4096, (_first + _cells) * SUPER2_H4096), t=SUPER2_T1,
                                              gamma=G14, boundary=OUTFLOW, cells_4096=_cells, first_4096=_first)


# A probe for the layout (LUNA only): stage S alone, the buffers either side within 0.9 of S, to measure how far S's waves reach.
_S = _V2["stages"]["S"]
PROBLEMS["super-v2-s-probe"] = dict(kind="super2", box=(_S[0] - 0.9, _S[1] + 0.9), t=SUPER2_T1, gamma=G14, boundary=OUTFLOW)


def super2_extras(a, b, t=0.0):
    """The smooth parts of v2 on cells [a, b] at t, conserved averages, exact while each stays in its own gas: stage B's waves (v1's,
    moved to SUPER2_B); stage M's bump, rho += 0.5 sin^2(pi (x - x0) / 0.5) on [x0, x0 + 0.5] at v = -11.83 and uniform P, carried at
    v; Shu-Osher's sine, rho += 0.2 sin(5 (x - x_s)) on [x_s, end] at rest (t = 0 only: the shock reaches it at once)."""
    e, sound = super_wave(a - (SUPER2_B - 3.0), b - (SUPER2_B - 3.0), t)
    c = np.sqrt(G14)
    U = SUPER_WAVE * (e * np.array([1.0, 0.0, 0.0])[:, None] + sound * np.array([1.0, c, G14 / (G14 - 1.0)])[:, None])
    v, w, x0 = -lay.V_MACH, lay.BUMP_W, lay.BUMP_X0 - lay.V_MACH * t
    lo, hi = np.clip(a, x0, x0 + w), np.clip(b, x0, x0 + w)
    prim = lambda s: 0.5 * s - w / (4.0 * np.pi) * np.sin(2.0 * np.pi * s / w)  # noqa: E731
    drho = lay.BUMP_AMP * (prim(hi - x0) - prim(lo - x0)) / (b - a)
    U = U + drho * np.array([1.0, v, 0.5 * v * v])[:, None]
    if t == 0.0:
        xs, xe = SUPER2_SO
        lo, hi = np.clip(a, xs, xe), np.clip(b, xs, xe)
        U = U + np.array([1.0, 0.0, 0.0])[:, None] * (lay.SO_AMP * (np.cos(lay.SO_K * (lo - xs)) - np.cos(lay.SO_K * (hi - xs))) / lay.SO_K / (b - a))
    return U


def super2_initial(a, b, m0=False):
    """The conserved cell averages of problems_1d.hpp super_tube_v2(): volume fractions of the segments plus super2_extras; m0: stage
    M's stream and bump replaced by the ambient gas (super-tube-v2-m0)."""
    segments = (((0.0, (1.0, 0.0, 1.0)),) + SUPER2_SEGMENTS[1:]) if m0 else SUPER2_SEGMENTS
    ends = [x for x, _ in segments[1:]] + [SUPER2_LENGTH]
    U = sum(np.clip(np.minimum(b, e) - np.maximum(a, x), 0.0, None) / (b - a) * conserved(*map(np.float64, W), G14)[:, None]
            for (x, W), e in zip(segments, ends))
    extras = super2_extras(a, b, 0.0)
    if m0:   # without the bump
        x0, w = lay.BUMP_X0, lay.BUMP_W
        lo, hi = np.clip(a, x0, x0 + w), np.clip(b, x0, x0 + w)
        prim = lambda s: 0.5 * s - w / (4.0 * np.pi) * np.sin(2.0 * np.pi * s / w)  # noqa: E731
        v = -lay.V_MACH
        extras = extras - lay.BUMP_AMP * (prim(hi - x0) - prim(lo - x0)) / (b - a) * np.array([1.0, v, 0.5 * v * v])[:, None]
    return U + extras


def riemann(WL, WR, xi, g=G14):
    """(rho, u, p) of the exact Riemann solution at xi = x / t (Toro 2009, ch. 4; vacuum generation, sec. 4.6) for pressures down to
    1e-30: the star state by bisection (super_tube_layout.star), gamma = 1.4."""
    if lay.vacuum(WL, WR):
        return vacuum_sample(WL, WR, g, xi)
    p, u = lay.star(WL, WR, 1e-30)
    out = [np.empty_like(xi) for _ in range(3)]
    for W, side, m in ((WL, -1.0, xi < u), (WR, 1.0, xi >= u)):
        r, v, pk = W
        ck = np.sqrt(g * pk / r)
        x = xi[m]
        if p > pk:   # shock
            s = v + side * ck * np.sqrt((g + 1) / (2 * g) * p / pk + (g - 1) / (2 * g))
            rs = r * ((p / pk + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pk + 1))
            outside = side * (x - s) > 0
            vals = [np.where(outside, r, rs), np.where(outside, v, u), np.where(outside, pk, p)]
        else:        # rarefaction: head at v + side c, tail at u + side c*
            cs = ck * (p / pk) ** ((g - 1) / (2 * g))
            head, tail = v + side * ck, u + side * cs
            cf = 2.0 / (g + 1) * (ck - side * (g - 1) / 2 * (v - x))
            uf = 2.0 / (g + 1) * (-side * ck + (g - 1) / 2 * v + x)
            cf = np.maximum(cf, 0.0)
            rf, pf = r * (cf / ck) ** (2 / (g - 1)), pk * (cf / ck) ** (2 * g / (g - 1))
            outside, inside = side * (x - head) > 0, side * (x - tail) <= 0
            vals = [np.where(outside, r, np.where(inside, r * (p / pk) ** (1 / g), rf)), np.where(outside, v, np.where(inside, u, uf)),
                    np.where(outside, pk, np.where(inside, p, pf))]
        for k in range(3):
            out[k][m] = vals[k]
    return tuple(out)


def super2_fans(t):
    """(x0, WL, WR, lo, hi) of every interface of v2's segments, [lo, hi] its waves at t (a contact alone, or equal states: lo = hi).
    Interfaces at or inside SO, S and H: lo = hi = 0 (they interact within; not exact)."""
    out = []
    inexact = [_V2["stages"][k] for k, *_ in lay.STAGES_V2 if k not in lay.EXACT_V2]
    for (_, WL), (x0, WR) in zip(SUPER2_SEGMENTS, SUPER2_SEGMENTS[1:]):
        skip = WL == WR or any(a <= x0 <= b for a, b in inexact)
        lo, hi = (0.0, 0.0) if skip else lay.fan(tuple(map(float, WL)), tuple(map(float, WR)))
        out.append((x0, WL, WR, x0 + lo * t, x0 + hi * t))
    return out


def super2_meets(t):
    """Pairs of neighbouring interfaces of v2 whose waves overlap at t, by x0, outside S and H (they interact within at once)."""
    fans = super2_fans(t)
    inside = [super2_region(k) for k in ("S", "H")]
    return [(A[0], B[0]) for A, B in zip(fans, fans[1:]) if A[4] > B[3] and not any(lo < A[0] < hi and lo < B[0] < hi for lo, hi in inside)]


def super2_exact(a, b, t=SUPER2_T1, sub=256):
    """(rho, u, p) averages of the cells at t <= 0.1 on v2's exact stages (super_tube_layout.EXACT_V2): each interface's Riemann
    solution inside its waves, between them the segment's state, plus super2_extras at t. SO, S and H cells: not exact."""
    f = (np.arange(sub) + 0.5) / sub
    out = np.empty((3, a.size))
    fans = super2_fans(t)
    his = np.array([hi for *_, hi in fans])
    states = [np.array([w[i] for _, w in SUPER2_SEGMENTS], float) for i in range(3)]
    for s in range(0, a.size, 4096):
        A, B = a[s:s + 4096], b[s:s + 4096]
        x = A[:, None] + (B - A)[:, None] * f[None, :]
        k = np.searchsorted(np.sort(his), x, side="right")     # the segment between two interfaces' waves
        W = [st[k] for st in states]
        for x0, WL, WR, lo, hi in fans:
            m = (x >= lo) & (x < hi)
            if m.any() and hi > lo:
                fan = riemann(tuple(map(float, WL)), tuple(map(float, WR)), (x[m] - x0) / t)
                for i in range(3):
                    W[i][m] = fan[i]
        out[:, s:s + 4096] = conserved(*W, G14).mean(axis=2)
    out += super2_extras(a, b, t)
    return primitive(out, G14)
