"""Solver 2D incompressible Navier-Stokes (laminar), grid staggered (MAC) dengan benda tertanam.
Metode: semi-Lagrangian advection + difusi eksplisit + proyeksi tekanan hanya di sel fluida (sparse LU).
Gaya: integral tekanan + geseran pada permukaan benda.
Chord = 1, kecepatan freestream = 1, jadi Re = 1/nu."""
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import splu

NX, NY, LX = 250, 120, 5.0
DX = LX / NX
LY = NY * DX
XLE = 1.0  # posisi leading edge di domain


def _shape(x, m, p, t):
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m == 0:
        return np.zeros_like(x), yt, np.zeros_like(x)
    fr = x < p
    yc = np.where(fr, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
    dy = np.where(fr, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, yt, dy


def airfoil_coords(m, p, t, n=80):
    b = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    yc, yt, dy = _shape(b, m, p, t)
    th = np.arctan(dy)
    xu, yu = b - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = b + yt * np.sin(th), yc - yt * np.cos(th)
    return np.r_[xu[::-1], xl[1:]], np.r_[yu[::-1], yl[1:]]


def normalize(coords):
    """Skala koordinat custom: leading edge di (0,0), chord = 1."""
    a = np.asarray(coords, float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 10 or not np.isfinite(a).all():
        raise ValueError("Koordinat airfoil tidak valid (minimal 10 titik x,y).")
    le = a[np.argmin(a[:, 0])]
    c = a[:, 0].max() - le[0]
    if c <= 0:
        raise ValueError("Chord airfoil tidak valid.")
    a = (a - le) / c
    if np.abs(a[:, 1]).max() > 0.5:
        raise ValueError("Bentuk airfoil tidak wajar (tebal/kamber terlalu besar).")
    return a[:, 0], a[:, 1]


def _inside(X, Y, px, py):
    """Point-in-polygon (ray casting) untuk semua pusat sel sekaligus."""
    ins = np.zeros(X.shape, bool)
    j = len(px) - 1
    for i in range(len(px)):
        cond = (py[i] > Y) != (py[j] > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (px[j] - px[i]) * (Y - py[i]) / (py[j] - py[i]) + px[i]
        ins ^= cond & (X < xint)
        j = i
    return ins


def _mask(m, p, t, coords=None):
    xs = (np.arange(NX) + 0.5) * DX - XLE
    ys = (np.arange(NY) + 0.5) * DX - LY / 2
    X, Y = np.meshgrid(xs, ys)
    if coords is not None:
        px, py = normalize(coords)
        return _inside(X, Y, px, py), xs, ys
    yc, yt, _ = _shape(np.clip(X, 0, 1), m, p, t)
    return ((X >= 0) & (X <= 1) & (np.abs(Y - yc) <= yt)), xs, ys


def _setup(solid):
    """Topologi MAC: muka u/v yang ikut proyeksi, muka yang dinolkan (dinding benda), dan matriks Poisson."""
    uo = np.zeros((NY, NX + 1), bool)
    uo[:, 1:NX] = ~(solid[:, :-1] | solid[:, 1:])
    uo[:, NX] = ~solid[:, -1]                      # outlet: p = 0
    vo = np.zeros((NY + 1, NX), bool)
    vo[1:NY, :] = ~(solid[:-1, :] | solid[1:, :])
    us = np.zeros((NY, NX + 1), bool); us[:, 1:NX] = solid[:, :-1] | solid[:, 1:]
    vs = np.zeros((NY + 1, NX), bool); vs[1:NY, :] = solid[:-1, :] | solid[1:, :]
    fl = ~solid
    idx = -np.ones((NY, NX), int)
    idx[fl] = np.arange(fl.sum())
    J, I = np.nonzero(fl)
    k = idx[J, I]
    diag = np.zeros(len(k)); R, C = [], []
    for op, nj, ni in ((uo[J, I + 1], J, I + 1), (uo[J, I], J, I - 1), (vo[J + 1, I], J + 1, I), (vo[J, I], J - 1, I)):
        diag += op
        m = op & (ni >= 0) & (ni < NX) & (nj >= 0) & (nj < NY)
        R.append(k[m]); C.append(idx[nj[m], ni[m]])
    R, C = np.concatenate(R), np.concatenate(C)
    A = (sp.coo_matrix((-np.ones(len(R)), (R, C)), shape=(len(k),) * 2) + sp.diags(diag)).tocsc()
    return splu(A), fl, uo, vo, us, vs


ADVECTION = "maccormack"   # "maccormack" (orde 2, dengan limiter) atau "semilagrangian" (orde 1)


def _mc(f, r, c):
    return map_coordinates(f, [r, c], order=1, mode="nearest")


def _adv(f, R, C, dr, dc):
    """Advection MacCormack + limiter: sampel di (R - dr, C - dc), koreksi galat dengan langkah balik."""
    r, c = R - dr, C - dc
    f1 = _mc(f, r, c)
    f2 = _mc(f1, R + dr, C + dc)
    out = f1 + 0.5 * (f - f2)
    r0 = np.clip(np.floor(r).astype(int), 0, f.shape[0] - 2)
    c0 = np.clip(np.floor(c).astype(int), 0, f.shape[1] - 2)
    nb = (f[r0, c0], f[r0 + 1, c0], f[r0, c0 + 1], f[r0 + 1, c0 + 1])
    return np.clip(out, np.minimum.reduce(nb), np.maximum.reduce(nb))


def _lap(f):
    g = np.pad(f, 1, mode="edge")
    return (g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2] - 4 * f) / DX**2


def _force(solid, pr, u, v, nu):
    """Gaya pada benda = integral (-p n + geseran) di muka sel padat yang berbatasan dengan sel fluida."""
    R_ = solid[:, :-1] & ~solid[:, 1:]      # padat di kiri, fluida di kanan
    L_ = ~solid[:, :-1] & solid[:, 1:]      # fluida di kiri, padat di kanan
    T_ = solid[:-1, :] & ~solid[1:, :]      # padat di bawah, fluida di atas
    B_ = ~solid[:-1, :] & solid[1:, :]      # fluida di bawah, padat di atas
    fx = DX * (-pr[:, 1:][R_].sum() + pr[:, :-1][L_].sum())
    fy = DX * (-pr[1:, :][T_].sum() + pr[:-1, :][B_].sum())
    vc_r = 0.5 * (v[:-1, 1:] + v[1:, 1:]); vc_l = 0.5 * (v[:-1, :-1] + v[1:, :-1])
    uc_t = 0.5 * (u[1:, :-1] + u[1:, 1:]); uc_b = 0.5 * (u[:-1, :-1] + u[:-1, 1:])
    fy += 2 * nu * (vc_r[R_].sum() + vc_l[L_].sum())      # geseran: dx * nu * (u_t / (dx/2))
    fx += 2 * nu * (uc_t[T_].sum() + uc_b[B_].sum())
    return fx, fy


def simulate(m, p, t, alpha, Re, steps=800, nframes=20, coords=None):
    solid, xs, ys = _mask(m, p, t, coords)
    if not solid.any():
        raise ValueError("Airfoil terlalu tipis untuk resolusi grid ini.")
    lu, fl, uo, vo, us, vs = _setup(solid)
    dt, nu = 0.4 * DX, 1.0 / Re
    a = np.radians(alpha); ua, va = np.cos(a), np.sin(a)
    u = np.full((NY, NX + 1), ua); v = np.full((NY + 1, NX), va)
    u[us] = 0; v[vs] = 0
    Au, Bu = np.mgrid[0:NY, 0:NX + 1].astype(float)
    Av, Bv = np.mgrid[0:NY + 1, 0:NX].astype(float)
    mc = lambda f, r, c: map_coordinates(f, [r, c], order=1, mode="nearest")
    every = max(1, steps // nframes)
    speed, vort, cl, cd = [], [], [], []

    for s in range(steps):
        vu = mc(v, Au + 0.5, Bu - 0.5); uv = mc(u, Av - 0.5, Bv + 0.5)      # silang antar-muka
        if ADVECTION == "maccormack":
            u, v = _adv(u, Au, Bu, vu * dt / DX, u * dt / DX), _adv(v, Av, Bv, v * dt / DX, uv * dt / DX)
        else:
            u, v = mc(u, Au - vu * dt / DX, Bu - u * dt / DX), mc(v, Av - v * dt / DX, Bv - uv * dt / DX)
        u = u + nu * dt * _lap(u); v = v + nu * dt * _lap(v)
        u[:, 0] = ua; v[:, 0] = va; v[0, :] = va; v[NY, :] = va
        u[:, NX] = u[:, NX - 1]; v[:, -1] = v[:, -2]
        u[us] = 0; v[vs] = 0
        div = (u[:, 1:] - u[:, :-1] + v[1:, :] - v[:-1, :]) / DX
        pr = np.zeros((NY, NX))
        pr[fl] = lu.solve(-(DX * DX / dt) * div[fl])
        gu = np.zeros_like(u); gu[:, 1:NX] = pr[:, 1:] - pr[:, :-1]; gu[:, NX] = -pr[:, -1]
        gv = np.zeros_like(v); gv[1:NY, :] = pr[1:, :] - pr[:-1, :]
        u = u - dt * gu * uo / DX
        v = v - dt * gv * vo / DX
        fx, fy = _force(solid, pr, u, v, nu)
        cl.append(2 * (-fx * np.sin(a) + fy * np.cos(a)))
        cd.append(2 * (fx * np.cos(a) + fy * np.sin(a)))
        if s % every == every - 1:
            uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1] + v[1:])
            if not np.isfinite(uc).all():
                raise ValueError("Simulasi tidak stabil, coba Re lebih rendah.")
            speed.append(np.hypot(uc, vc)[::2, ::2].round(2).tolist())
            w = np.gradient(vc, DX, axis=1) - np.gradient(uc, DX, axis=0)
            vort.append(w[::2, ::2].round(1).tolist())

    tail = slice(int(steps * 0.7), None)
    sm = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")[::5].round(3).tolist()
    return {
        "x": xs[::2].round(3).tolist(), "y": ys[::2].round(3).tolist(),
        "mask": solid[::2, ::2].astype(int).tolist(),
        "speed": speed, "vort": vort,
        "hist_cl": sm(cl), "hist_cd": sm(cd),
        "Cl": float(np.mean(cl[tail])), "Cd": float(np.mean(cd[tail])),
    }