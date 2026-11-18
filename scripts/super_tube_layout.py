#!/usr/bin/env python3
"""Layout and wave timing of the super-hard 1D tube: exact Riemann fronts of every interface, and when stages first meet.

    python super_tube_layout.py        # v1 (research_notes/super_tube_2026-11-10.md)
    python super_tube_layout.py v2     # v2 (research_notes/super_tube_v2_2026-11-18.md): stages, buffers, times, segment tables

Plain Python, no numpy: scripts/problems_1d.py imports v2() for its segment table; src/hydro/problems_1d.hpp holds the same numbers.
"""
import math
import sys

g = 1.4


def c(r, p): return math.sqrt(g * p / r)


def f(p, r, pk):
    A = 2 / ((g + 1) * r); B = (g - 1) / (g + 1) * pk; ck = c(r, pk)
    if p > pk: return (p - pk) * math.sqrt(A / (p + B))
    return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)


def vacuum(L, R):
    """Whether the Riemann problem generates vacuum: u_R - u_L >= 2 (c_L + c_R) / (gamma - 1) (Toro sec. 4.6)."""
    return 2 * (c(L[0], L[2]) + c(R[0], R[2])) / (g - 1) <= R[1] - L[1]


def star(L, R, lo=1e-14):
    (rl, ul, pl), (rr, ur, pr) = L, R
    hi = 1e14
    for _ in range(300):
        p = math.sqrt(lo * hi)
        v = f(p, rl, pl) + f(p, rr, pr) + ur - ul
        if v > 0: hi = p
        else: lo = p
    p = math.sqrt(lo * hi); u = 0.5 * (ul + ur) + 0.5 * (f(p, rr, pr) - f(p, rl, pl)); return p, u


def speeds(L, R, lo=1e-14):
    (rl, ul, pl), (rr, ur, pr) = L, R
    if vacuum(L, R):   # two rarefactions into vacuum: heads at u -+ c, tails at the escape speeds
        al, ar = c(rl, pl), c(rr, pr)
        return dict(p=0.0, u=float('nan'), left=('rarefaction', ul - al, ul + 2 * al / (g - 1)),
                    right=('rarefaction', ur - 2 * ar / (g - 1), ur + ar), rl=0.0, rr=0.0)
    p, u = star(L, R, lo)
    if p > pl: sl = ul - c(rl, pl) * math.sqrt((g + 1) / (2 * g) * p / pl + (g - 1) / (2 * g)); left = ('shock', sl, sl)
    else:
        cs = c(rl, pl) * (p / pl) ** ((g - 1) / (2 * g)); left = ('rarefaction', ul - c(rl, pl), u - cs)
    if p > pr: sr = ur + c(rr, pr) * math.sqrt((g + 1) / (2 * g) * p / pr + (g - 1) / (2 * g)); right = ('shock', sr, sr)
    else:
        cs = c(rr, pr) * (p / pr) ** ((g - 1) / (2 * g)); right = ('rarefaction', u + cs, ur + c(rr, pr))

    def rstar(r, pk, side):
        if p > pk: return r * ((p / pk + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pk + 1))
        return r * (p / pk) ** (1 / g)
    return dict(p=p, u=u, left=left, right=right, rl=rstar(rl, pl, 'l'), rr=rstar(rr, pr, 'r'))


amb = (1.0, 0.0, 1.0); mv = (1.0, 1.0, 1.0)
stages = [('A quiet', [(1.0, amb)]),
          ('B smooth waves', [(1.0, amb)]),
          ('C stationary peak', [(0.05, (10.0, 0.0, 1.0))]),
          ('C2 moving peak', [(0.75, mv), (0.05, (10.0, 1.0, 1.0)), (0.75, mv)]),
          ('D sonic rarefaction', [(0.5, (1.0, 0.75, 1.0)), (0.5, (0.125, 0.0, 0.1))]),
          ('E extreme jump', [(0.5, (1.0, 0.0, 0.1)), (0.5, (1e-3, 0.0, 1e-10))]),
          ('F slow shock', [(0.5, (3.86, -0.81, 10.33)), (0.5, (1.0, -3.44, 1.0))]),
          ('S colliding streams', [(0.3, (5.99924, 19.5975, 460.894)), (0.3, (5.99242, -6.19633, 46.0950))]),
          ('G near vacuum', [(0.5, (1.0, -2.0, 0.4)), (0.5, (1.0, 2.0, 0.4))]),
          ('H double blast', [(0.1, (1.0, 0.0, 1000.0)), (0.8, (1.0, 0.0, 0.01)), (0.1, (1.0, 0.0, 100.0))])]


def layout(bufs):
    segs = []; x = 0.0
    for i, (n, parts) in enumerate(stages):
        if i > 0: segs.append((x, 'buffer', amb)); x += bufs[i - 1]
        for w, s in parts: segs.append((x, n, s)); x += w
    return segs, x


def fronts(segs):
    fr = []
    for (x, a, L), (x2, b, R) in zip(segs, segs[1:]):
        if L == R: continue
        s = speeds(L, R); fr.append((x2, min(s['left'][1], s['left'][2]), max(s['right'][1], s['right'][2]), a, b))
    return fr


def meets(fr):
    out = []
    for A, B in zip(fr, fr[1:]):
        if A[2] > B[1] and not (A[3].startswith('H') and B[3].startswith('H')) and not (A[4] == B[3] == A[3] or A[3] == B[4]):
            out.append(((B[0] - A[0]) / (A[2] - B[1]), A[3], A[4], B[3], B[4]))
    return sorted(out)


def v1():
    # buffers sized per neighbour speed: start uniform 2, then widen near fast stages
    bufs = [2.0] * (len(stages) - 1)
    names = [s[0] for s in stages]
    bufs[names.index('S colliding streams') - 1] = 4.0
    bufs[names.index('S colliding streams')] = 4.0
    bufs[names.index('H double blast') - 1] = 4.0
    segs, end = layout(bufs); fr = fronts(segs)
    print('length', round(end, 2))
    for m in meets(fr)[:8]: print(f"t={m[0]:.4f}  {m[1][:20]}|{m[2][:20]}  meets  {m[3][:20]}|{m[4][:20]}")
    for x, l, r, a, b in fr: print(f"{x:6.2f} {l:8.2f} {r:8.2f}  {a[:22]} | {b[:22]}")


# ---- v2 ---------------------------------------------------------------------------------------------------------------------
# Mach 1e4 at rho = 1, P = 1e-6: v = 1e4 c = 11.832; the stream runs left, out through the open wall.
MACH = 1e4
V_MACH = MACH * math.sqrt(1.4 * 1e-6)
SO_POST = (3.857143, 2.629369, 10.33333)     # Shu-Osher: behind a Mach 3 shock into (1, 0, 1)
SO_K, SO_AMP, SO_PERIODS = 5.0, 0.2, 3       # rho = 1 + 0.2 sin(5 (x - x_shock)) ahead of the shock, 3 wavelengths
BUMP_AMP, BUMP_W, BUMP_X0 = 0.5, 0.5, 1.3    # stage M: rho = 1 + 0.5 sin^2(pi (x - 1.3) / 0.5) on [1.3, 1.8]; at T1 [0.12, 0.62]
T1, T_INDEP, MIN_BUFFER = 0.1, 0.15, 1.0     # checkpoint; buffers keep stages apart to T_INDEP at least; and at least 1 wide

# (key, name, parts [(width, (rho, v, P))], what it tests). Parts within a stage are Riemann problems with each other.
STAGES_V2 = [
    ('M', 'Mach 1e4 advection', [(2.0, (1.0, -V_MACH, 1e-6))], 'density bump in cold gas at Mach 1e4, out through the open wall'),
    ('A', 'quiet', [(1.0, amb)], 'noise from mesh motion alone'),
    ('B', 'smooth waves', [(1.0, amb)], 'order on smooth flow'),
    ('C', 'stationary peak', [(0.05, (10.0, 0.0, 1.0))], 'contact sharpness, spurious pressure'),
    ('C2', 'moving peak', [(0.75, mv), (0.05, (10.0, 1.0, 1.0)), (0.75, mv)], 'moving contact'),
    ('D', 'sonic rarefaction', [(0.5, (1.0, 0.75, 1.0)), (0.5, (0.125, 0.0, 0.1))], 'entropy glitch at the sonic point'),
    ('E', 'extreme jump', [(0.5, (1.0, 0.0, 0.1)), (0.5, (1e-3, 0.0, 1e-10))], 'shock position, heating at the contact'),
    ('L', 'LeBlanc-type jump', [(0.5, (1.0, 0.0, 0.1)), (1.0, (1e-6, 0.0, 1e-13))], 'density ratio 1e6, pressure ratio 1e12'),
    ('F', 'slow shock', [(0.5, (3.857143, -0.810631, 10.33333)), (0.5, (1.0, -3.44, 1.0))], 'ripples behind a slow shock'),
    ('SO', 'Shu-Osher', [(1.0, SO_POST), (3.8, amb)], 'Mach 3 shock into a density sine (on the first 6 pi / 5 of the 3.8)'),
    ('N', 'Noh streams', [(0.75, (1.0, 1.0, 1e-6)), (0.75, (1.0, -1.0, 1e-6))], 'cold colliding streams: wall heating'),
    ('S', 'colliding streams', [(0.3, (5.99924, 19.5975, 460.894)), (0.3, (5.99242, -6.19633, 46.0950))], 'strongest shocks from velocity'),
    ('G', 'near vacuum', [(0.5, (1.0, -2.0, 0.4)), (0.5, (1.0, 2.0, 0.4))], 'positivity, density down to about 0.02'),
    ('V', 'true vacuum', [(0.75, (1.0, -4.0, 0.4)), (0.75, (1.0, 4.0, 0.4))], 'vacuum: |dv| = 8 > 2 (cL + cR) / (gamma - 1) = 7.48'),
    ('H', 'double blast', [(0.1, (1.0, 0.0, 1000.0)), (0.8, (1.0, 0.0, 0.01)), (0.1, (1.0, 0.0, 100.0))], 'collisions and reflections'),
]
EXACT_V2 = ('M', 'A', 'B', 'C', 'C2', 'D', 'E', 'L', 'F', 'N', 'G', 'V')   # exact at T1: Riemann superposition (B linear, M translated)
# Stages whose waves interact inside at once send faster waves out than their edge fronts: the speed of their outermost disturbance
# to the right, measured. S: Toro 5's right shock leaves its gas into the ambient at about 15 (v1, LUNA 2^20: 1.49 past S's
# right edge at t = 0.1); 16 with margin.
SPREAD_RIGHT = {'S': 16.0}
# and to the left: S alone (super-v2-s-probe, LUNA, 2^17 spacing) is disturbed 0.21 left of its edge at t = 0.05, 0.84 at 0.1 (a wave
# at about 12.6 from its inner collision): 11 t bounds both and the extrapolated 1.47 at 0.15.
SPREAD_LEFT = {'S': 11.0}
INTERACTING = ('S', 'H')   # their inner fronts' speeds say nothing about how far they reach: their edge fronts and SPREAD_RIGHT only


def fan(L, R):
    """(lo, hi) speeds of the waves of an interface; a side whose pressure stays has no wave (a contact: lo = hi = u)."""
    s = speeds(L, R, 1e-30)
    if s['p'] == 0.0:
        return s['left'][1], s['right'][2]
    lo = min(s['left'][1:]) if abs(s['p'] / L[2] - 1) > 1e-9 else s['u']
    hi = max(s['right'][1:]) if abs(s['p'] / R[2] - 1) > 1e-9 else s['u']
    return lo, hi


def v2_segments(bufs):
    """[(x, stage key, state)], the length: stages from the open wall at 0, buffers of the ambient state between."""
    segs, x = [], 0.0
    for i, (k, _, parts, _) in enumerate(STAGES_V2):
        if i > 0:
            segs.append((x, 'buffer', amb)); x = round(x + bufs[i - 1], 10)
        for w, s in parts:
            segs.append((x, k, s)); x = round(x + w, 10)   # decimal positions: the same double in C++ and in exported ICs
    return segs, x


def v2_fronts(segs):
    out = []
    for (_, a, L), (x, b, R) in zip(segs, segs[1:]):
        if L != R:
            lo, hi = fan(L, R)
            out.append((x, lo, hi, a, b))
    return out


def v2_meet(A, B):
    return (B[0] - A[0]) / (A[2] - B[1]) if A[2] > B[1] else math.inf


def v2(T=T_INDEP, min_buffer=MIN_BUFFER):
    """Buffers so that no two stages' waves meet before T (each at least min_buffer, rounded up to 0.25), from the fronts of the
    stage edges; returns dict(segments, length, stages {key: (x0, x1)}, fronts, buffers)."""
    bufs = [min_buffer] * (len(STAGES_V2) - 1)
    for _ in range(4):   # the edge fronts do not depend on the buffer widths; one pass sets them, the rest confirm
        segs, end = v2_segments(bufs)
        fr = v2_fronts(segs)
        for i in range(len(bufs)):
            left = [f for f in fr if f[3] == STAGES_V2[i][0] and f[4] == 'buffer']
            right = [f for f in fr if f[3] == 'buffer' and f[4] == STAGES_V2[i + 1][0]]
            out = max(left[0][2], SPREAD_RIGHT.get(STAGES_V2[i][0], -math.inf)) if left else 0.0
            into = min(right[0][1], -SPREAD_LEFT.get(STAGES_V2[i + 1][0], -math.inf)) if right else 0.0
            need = (out - into) * T if left and right else 0.0
            bufs[i] = max(min_buffer, math.ceil(need / 0.25 - 1e-9) * 0.25)
    segs, end = v2_segments(bufs)
    fr = v2_fronts(segs)
    cuts = [0.0]   # between stages: the middle of the quiet line at T1 (each stage's fronts, S's measured spread)
    for (k, *_), (k2, *_r) in zip(STAGES_V2, STAGES_V2[1:]):
        last = max(x for x, kk, _ in segs if kk == k)
        stage_end = next(x for x, kk, _ in segs if x > last)
        right = [f for f in fr if f[3:] == (k, 'buffer') or (k not in INTERACTING and k in f[3:])]
        reach = max([x + hi * T1 for x, lo, hi, a, b in right] + [stage_end + SPREAD_RIGHT.get(k, 0.0) * T1])
        start = next(x for x, kk, _ in segs if kk == k2)
        left = [f for f in fr if f[3:] == ('buffer', k2) or (k2 not in INTERACTING and k2 in f[3:])]
        reach2 = min([x + lo * T1 for x, lo, hi, a, b in left] + [start - SPREAD_LEFT.get(k2, 0.0) * T1])
        cuts.append(round(0.5 * (reach + reach2), 4))
    cuts.append(end)
    xs = {}
    for x, k, _ in segs:
        if k != 'buffer':
            xs.setdefault(k, [x, x])
    for (x, k, _), (x2, *_r) in zip(segs, segs[1:] + [(end, None, None)]):
        if k != 'buffer':
            xs[k][1] = x2
    return dict(segments=segs, length=end, stages={k: tuple(v) for k, v in xs.items()}, fronts=fr, buffers=bufs,
                regions={k: (lo, hi) for (k, *_), lo, hi in zip(STAGES_V2, cuts, cuts[1:])})


def v2_report():
    d = v2()
    fr, st = d['fronts'], d['stages']
    print(f"length {d['length']:.4f}; buffers {d['buffers']}")
    print('\n| key | stage | where | scored on | parts (width, (rho, v, P)) | tests |\n|---|---|---|---|---|---|')
    for k, name, parts, tests in STAGES_V2:
        a, b = st[k]
        lo, hi = d['regions'][k]
        print(f"| {k} | {name} | {a:g} to {b:g} | {lo:g} to {hi:g} | " + ", ".join(f"{w:.4g}: ({s[0]:g}, {s[1]:.6g}, {s[2]:g})" for w, s in parts)
              + f" | {tests} |")
    print('\nfronts at t = 1 (lo, hi speeds):')
    for x, lo, hi, a, b in fr:
        print(f"  {x:8.4f} {lo:9.4f} {hi:9.4f}  {a} | {b}")
    pairs = [(v2_meet(A, B), A, B) for A, B in zip(fr, fr[1:])]
    across = sorted((t, A, B) for t, A, B in pairs if A[4] == 'buffer' and B[3] == 'buffer')
    print(f"\nfirst meetings across buffers: " + "; ".join(f"t = {t:.4f} {A[3]} with {B[4]}" for t, A, B in across[:5]))
    for k, s in SPREAD_RIGHT.items():
        nxt = next(B for A, B in zip(fr, fr[1:]) if A[3] == k and A[4] == 'buffer' and B[3] == 'buffer')
        print(f"  {k}'s measured spread ({s} to the right) reaches {nxt[4]} at t = {(nxt[0] - st[k][1]) / (s - nxt[1]):.4f}")
    for k, s in SPREAD_LEFT.items():
        prv = next(A for A, B in zip(fr, fr[1:]) if B[3] == 'buffer' and B[4] == k and A[4] == 'buffer')
        print(f"  {k}'s measured spread ({s} to the left) reaches {prv[3]} at t = {(st[k][0] - prv[0]) / (s + prv[2]):.4f}")
    for k, *_ in STAGES_V2:   # a stage's own fronts: the first meeting inside it (its edges with the buffers included)
        own = [f for f in fr if k in (f[3], f[4])]
        ts = [v2_meet(A, B) for A, B in zip(own, own[1:])]
        print(f"  stage {k:3s}: single Riemann problems until t = {min(ts) if ts else math.inf:.4f}")
    print('\nsegments (x, rho, v, P):')
    print(',\n'.join(f"    ({x:.10g}, ({s[0]!r}, {s[1]!r}, {s[2]!r}))" for x, _, s in d['segments']))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'v2':
        v2_report()
    else:
        v1()
