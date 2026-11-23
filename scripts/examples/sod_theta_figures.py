"""Figures and the comparison table of the Sod theta example: luna-cpp (hand-written tangent and adjoint) against LUNA (JAX).

    python sod_theta_figures.py <sod_theta_cpp.npz> <sod_theta_luna.npz> <out dir>

Writes sod_theta_fd.png (gradient against finite differences, with astronomix's published point), sod_theta_map.png (the final
state and the sensitivity map dJ/dU0 of both codes), sod_theta_inverse.png (the inverse problem's loss and parameter error per
gradient) and prints the numbers.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from figstyle import FIGSIZE, apply_style, decorate, legend  # noqa: E402

THETA_SOD = np.array([1.0, 1.0, 0.0, 0.125, 0.1, 0.0])
NAMES = [r"$\rho_L$", r"$P_L$", r"$u_L$", r"$\rho_R$", r"$P_R$", r"$u_R$"]
# astronomix (Storcks, Thuerey & Buck 2026, arXiv:2607.18176, sec. 5.2): closer one-sided residual at h = 1e-6
ASTRONOMIX = {"FV (VL2, PLM, HLLC)": 5.9e-6, "FD (WENO-Z)": 7.2e-6}


def main(cpp_path, luna_path, out):
    apply_style()
    c, j = np.load(cpp_path), np.load(luna_path)
    os.makedirs(out, exist_ok=True)
    print("J: luna-cpp %.15e  LUNA %.15e  rel %.1e" % (c["J"], j["J"], abs(c["J"] - j["J"]) / abs(j["J"])))
    gscale = np.max(np.abs(j["g_adjoint"]))
    print("dJ/dtheta, luna-cpp adjoint / tangent, LUNA grad / jacfwd:")
    for k in range(6):
        print(f"  {k}: {c['g_adjoint'][k]: .12e} {c['g_tangent'][k]: .12e} {j['g_adjoint'][k]: .12e} {j['g_tangent'][k]: .12e}")
    print("largest differences / largest entry: cpp adj-tan %.1e, LUNA grad-jacfwd %.1e, cpp-LUNA %.1e" % (
        np.max(np.abs(c["g_adjoint"] - c["g_tangent"])) / gscale, np.max(np.abs(j["g_adjoint"] - j["g_tangent"])) / gscale,
        np.max(np.abs(c["g_adjoint"] - j["g_adjoint"])) / gscale))
    for i, v, ours, fc, fp, fm in c["map_checks"]:
        theirs = j["dJdU0"][int(i), int(v)]
        print(f"  map cell {int(i):3d} var {int(v)}: luna-cpp {ours: .6e}  LUNA {theirs: .6e}  FD central {fc: .6e} +h {fp: .6e} -h {fm: .6e}")
    mscale = np.max(np.abs(j["dJdU0"]))
    print("map dJ/dU0, cpp - LUNA: %.1e of the largest entry" % (np.max(np.abs(c["dJdU0"] - j["dJdU0"])) / mscale))

    # 1. FD sweep
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(c["hs"], c["fd_onesided"], "o-", color="C0", label="luna-cpp, closer one-sided")
    ax.loglog(c["hs"], c["fd_central"], "o--", color="C0", mfc="none", label="luna-cpp, central")
    ax.loglog(j["hs"], j["fd_onesided"], "s-", color="C1", ms=4, label="LUNA (JAX), closer one-sided")
    ax.loglog(j["hs"], j["fd_central"], "s--", color="C1", ms=4, mfc="none", label="LUNA (JAX), central")
    for (name, v), m in zip(ASTRONOMIX.items(), ["*", "P"]):
        ax.loglog([1e-6], [v], m, color="k", ms=11, label=f"astronomix {name}, published")
    h = np.array(c["hs"])
    ax.loglog(h, 3 * h, ":", color="0.4", label=r"$\propto h$")
    ax.set_xlabel(r"finite-difference step $h$")
    ax.set_ylabel(r"$\|g - g_{\rm FD}\| \,/\, \|g\|$")
    decorate(ax)
    legend(ax, fontsize=10, loc="upper left")
    fig.savefig(f"{out}/sod_theta_fd.png")
    plt.close(fig)

    # 2. final state and sensitivity map
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIGSIZE[0], FIGSIZE[1]))
    for key, col in [("rho", "C0"), ("u", "C1"), ("P", "C2")]:
        axs[0].plot(c["x"], c[key], "-", color=col, label={"rho": r"$\rho$", "u": r"$u$", "P": r"$P$"}[key] + " luna-cpp")
        axs[0].plot(j["x"], j[key], ".", color=col, ms=3, label="LUNA")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"state at $t = 0.2$")
    labels = [r"$\delta J/\delta \rho_0$", r"$\delta J/\delta (\rho u)_0$", r"$\delta J/\delta E_0$"]
    for v in range(3):
        axs[1].plot(c["x"], c["dJdU0"][:, v], "-", color=f"C{v}", label=labels[v] + " luna-cpp")
        axs[1].plot(j["x"], j["dJdU0"][:, v], ".", color=f"C{v}", ms=3, label="LUNA")
    ch = c["map_checks"]
    xc = c["x"][ch[:, 0].astype(int)]
    closer = np.where(np.abs(ch[:, 4] - ch[:, 2]) < np.abs(ch[:, 5] - ch[:, 2]), ch[:, 4], ch[:, 5])
    axs[1].plot(xc, closer, "kx", ms=8, label="closer one-sided difference")
    axs[1].set_xlabel(r"$x$ (initial)")
    axs[1].set_ylabel("sensitivity per unit length")
    # The two end cells (outflow walls copy them into the ghosts) are far larger: off the scale, written in.
    lim = 1.3 * np.max(np.abs(c["dJdU0"][2:-2]))
    axs[1].set_ylim(-lim, lim)
    ends = ", ".join(f"{v:.0f}" for v in np.concatenate([c["dJdU0"][0], c["dJdU0"][-1]]))
    axs[1].text(0.5, 0.04, f"end cells off scale: ({ends})", transform=axs[1].transAxes, ha="center", fontsize=10)
    for a in axs:
        decorate(a)
        legend(a, fontsize=9, ncol=2)
    fig.savefig(f"{out}/sod_theta_map.png")
    plt.close(fig)

    # 3. inverse problem
    fig, axs = plt.subplots(1, 2, figsize=(2 * FIGSIZE[0], FIGSIZE[1]))
    for d, col, name in [(c, "C0", "luna-cpp"), (j, "C1", "LUNA (JAX)")]:
        H = d["histories"]
        for k in range(H.shape[0]):
            L = H[k, :, 0]
            err = np.max(np.abs(H[k, :, 1:] - THETA_SOD), axis=1)
            ls = "-" if name == "luna-cpp" else "--"
            axs[0].semilogy(np.arange(1, len(L) + 1), np.minimum.accumulate(np.nan_to_num(L, nan=np.inf)), ls, color=col, lw=1.0,
                            label=name if k == 0 else None)
            axs[1].semilogy(np.arange(1, len(L) + 1), err, ls, color=col, lw=1.0, label=name if k == 0 else None)
    axs[0].set_ylabel(r"loss $L$ (best so far)")
    axs[1].set_ylabel(r"$\max_k |\theta_k - \theta_k^{\rm true}|$")
    for a in axs:
        a.set_xlabel("gradient evaluations (L-BFGS-B)")
        decorate(a)
        legend(a)
    fig.savefig(f"{out}/sod_theta_inverse.png")
    plt.close(fig)

    for d, name in [(c, "luna-cpp"), (j, "LUNA")]:
        e = np.max(np.abs(d["finals"] - THETA_SOD), axis=1)
        print(f"{name}: inverse, |theta - true| per start {np.array2string(e, precision=1)}, gradients {d['gradients']}, "
              f"wall {float(d['inversion_wall_s']):.1f} s")
    print("times (ms): luna-cpp forward+tape %.2f, adjoint %.2f, six tangents %.2f; LUNA forward %.2f, grad %.2f, jacfwd %.2f, peak %.1f MiB" % (
        c["ms_forward_tape"], c["ms_adjoint"], c["ms_six_tangents"], j["ms_forward"], j["ms_grad"], j["ms_jacfwd"], j["peak_bytes"] / 2**20))


if __name__ == "__main__":
    main(*sys.argv[1:4])
