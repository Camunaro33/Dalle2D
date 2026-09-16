# -*- coding: utf-8 -*-
"""
Streamlit applet version of SLAB2D_single_v1.py
Corner/point-supported rectangular slab, Kirchhoff thin-plate FE (ACM 12-DOF).

Run locally:   streamlit run app.py
Deploy:        push this file + requirements.txt to the GitHub repo,
                Streamlit Cloud auto-redeploys on every push.
"""

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st

st.set_page_config(page_title="Dalle 2D - GCI2011", layout="wide")

# ============================================================================
# CORE ENGINE (unchanged math from SLAB2D_single_v1.py) ---------------------
# ============================================================================

class PlateMesh:
    """Regular rectangular mesh, nx x ny elements over Lx x Ly."""
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        xs = np.linspace(0, Lx, nx + 1)
        ys = np.linspace(0, Ly, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        self.nodes = np.column_stack([X.ravel(order='C'), Y.ravel(order='C')])
        self.node_id = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
        elems = []
        for i in range(nx):
            for j in range(ny):
                n1 = self.node_id[i, j]
                n2 = self.node_id[i + 1, j]
                n3 = self.node_id[i + 1, j + 1]
                n4 = self.node_id[i, j + 1]
                elems.append([n1, n2, n3, n4])
        self.elems = np.array(elems)
        self.n_nodes = self.nodes.shape[0]

    def nearest_node(self, x, y):
        d2 = (self.nodes[:, 0] - x) ** 2 + (self.nodes[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def nearest_nodes(self, points):
        ids = [self.nearest_node(x, y) for (x, y) in points]
        snapped = [tuple(self.nodes[i]) for i in ids]
        return ids, snapped


@st.cache_resource(show_spinner="Dérivation symbolique de l'élément ACM (une seule fois)...")
def build_acm_element():
    x, y, a, b, nu_s, D_s, q_s = sp.symbols('x y a b nu D q')
    monoms = [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**3*y, x*y**3]
    P = sp.Matrix([monoms])

    def dP(ox, oy):
        return sp.Matrix([[sp.diff(m, x, ox, y, oy) for m in monoms]])

    node_xy = [(0, 0), (a, 0), (a, b), (0, b)]
    rows = []
    for (xi, yi) in node_xy:
        rows.append(P.subs({x: xi, y: yi}))
        rows.append(dP(1, 0).subs({x: xi, y: yi}))
        rows.append(dP(0, 1).subs({x: xi, y: yi}))
    C = sp.Matrix.vstack(*rows)
    Cinv = C.inv()

    N = P * Cinv
    Bx = dP(2, 0) * Cinv
    By = dP(0, 2) * Cinv
    Bxy = 2 * dP(1, 1) * Cinv
    Bmat = sp.Matrix.vstack(Bx, By, Bxy)

    Db = D_s * sp.Matrix([[1, nu_s, 0], [nu_s, 1, 0], [0, 0, (1 - nu_s) / 2]])

    Ke_sym = (Bmat.T * Db * Bmat).applyfunc(
        lambda e: sp.integrate(sp.integrate(e, (x, 0, a)), (y, 0, b)))
    Fe_sym = (N.T * q_s).applyfunc(
        lambda e: sp.integrate(sp.integrate(e, (x, 0, a)), (y, 0, b)))

    Ke_func = sp.lambdify((a, b, D_s, nu_s), Ke_sym, 'numpy')
    Fe_func = sp.lambdify((a, b, q_s), Fe_sym, 'numpy')
    return Ke_func, Fe_func


@st.cache_resource(show_spinner="Dérivation des opérateurs de moments (une seule fois)...")
def build_moment_recovery_ops():
    x, y, a, b = sp.symbols('x y a b')
    monoms = [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**3*y, x*y**3]
    P = sp.Matrix([monoms])

    def dP(ox, oy):
        return sp.Matrix([[sp.diff(m, x, ox, y, oy) for m in monoms]])

    node_xy = [(0, 0), (a, 0), (a, b), (0, b)]
    rows = []
    for (xi, yi) in node_xy:
        rows.append(P.subs({x: xi, y: yi}))
        rows.append(dP(1, 0).subs({x: xi, y: yi}))
        rows.append(dP(0, 1).subs({x: xi, y: yi}))
    C = sp.Matrix.vstack(*rows)
    Cinv = C.inv()

    ops = {
        'xx': dP(2, 0) * Cinv, 'yy': dP(0, 2) * Cinv, 'xy': dP(1, 1) * Cinv,
        'xxx': dP(3, 0) * Cinv, 'xyy': dP(1, 2) * Cinv,
        'yyy': dP(0, 3) * Cinv, 'xxy': dP(2, 1) * Cinv,
    }
    return {k: sp.lambdify((x, y, a, b), v, 'numpy') for k, v in ops.items()}


_Ke_func, _Fe_func = build_acm_element()
_mom_ops = build_moment_recovery_ops()


def recover_moments(mesh, U, D, nu):
    ex = mesh.Lx / mesh.nx
    ey = mesh.Ly / mesh.ny
    n_nodes = mesh.n_nodes
    corner_local = [(0, 0), (ex, 0), (ex, ey), (0, ey)]

    sums = {k: np.zeros(n_nodes) for k in ['mxx', 'myy', 'mxy', 'tx', 'ty']}
    counts = np.zeros(n_nodes)

    for el in mesh.elems:
        edofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in el]).ravel()
        d_e = U[edofs]
        for local_i, node in enumerate(el):
            xl, yl = corner_local[local_i]
            Wxx = float(np.array(_mom_ops['xx'](xl, yl, ex, ey)).flatten() @ d_e)
            Wyy = float(np.array(_mom_ops['yy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxy = float(np.array(_mom_ops['xy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxxx = float(np.array(_mom_ops['xxx'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxyy = float(np.array(_mom_ops['xyy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wyyy = float(np.array(_mom_ops['yyy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxxy = float(np.array(_mom_ops['xxy'](xl, yl, ex, ey)).flatten() @ d_e)

            sums['mxx'][node] += -D * (Wxx + nu * Wyy)
            sums['myy'][node] += -D * (Wyy + nu * Wxx)
            sums['mxy'][node] += -D * (1 - nu) * Wxy
            sums['tx'][node] += -D * (1 - nu) * (Wxxx + Wxyy)
            sums['ty'][node] += -D * (1 - nu) * (Wyyy + Wxxy)
            counts[node] += 1

    for k in sums:
        sums[k] /= counts
    return sums['mxx'], sums['myy'], sums['mxy'], sums['tx'], sums['ty']


def principal_moments(mxx, myy, mxy):
    avg = (mxx + myy) / 2.0
    R = np.sqrt(((mxx - myy) / 2.0) ** 2 + mxy ** 2)
    M1 = avg + R
    M2 = avg - R
    alpha = 0.5 * np.arctan2(2 * mxy, (mxx - myy))
    return M1, M2, alpha


def solve_plate_acm(mesh, D, nu, q, supported_nodes, point_loads=None):
    ex = mesh.Lx / mesh.nx
    ey = mesh.Ly / mesh.ny
    Ke = np.array(_Ke_func(ex, ey, D, nu), dtype=float)
    Fe = np.array(_Fe_func(ex, ey, q), dtype=float).flatten()

    ndof = 3 * mesh.n_nodes
    K = sparse.lil_matrix((ndof, ndof))
    F = np.zeros(ndof)

    for el in mesh.elems:
        edofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in el]).ravel()
        K[np.ix_(edofs, edofs)] += Ke
        F[edofs] += Fe

    if point_loads:
        for (xp, yp, P) in point_loads:
            node = mesh.nearest_node(xp, yp)
            F[3 * node] += P

    K = K.tocsr()
    fixed = np.array([3 * n for n in supported_nodes], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)

    Uf = spsolve(K[np.ix_(free, free)].tocsc(), F[free])
    U = np.zeros(ndof)
    U[free] = Uf
    return U[0::3], U


def As_required(M_field, fy, phi, As_min, d_eff):
    As = np.abs(M_field) / (phi * fy * 0.9 * d_eff)
    As = np.maximum(As, As_min)
    return As * 1e6  # mm^2/m


# ============================================================================
# SIDEBAR INPUTS --------------------------------------------------------------
# ============================================================================

st.sidebar.header("Géométrie & matériau")
L = st.sidebar.number_input("Portée L [m]", value=4.0, min_value=1.0, step=0.5)
t = st.sidebar.number_input("Épaisseur t [m]", value=0.20, min_value=0.05, step=0.01, format="%.3f")
E_GPa = st.sidebar.number_input("Module E [GPa]", value=25.0, min_value=1.0, step=1.0)
nu = st.sidebar.number_input("Coeff. Poisson ν", value=0.20, min_value=0.0, max_value=0.49, step=0.01)
E = E_GPa * 1e9

st.sidebar.header("Charges")
q_kPa = st.sidebar.slider("Charge uniforme q [kPa]", min_value=0.0, max_value=50.0,
                           value=10.0, step=0.5)
q = q_kPa * 1e3

use_point_load = st.sidebar.checkbox("Ajouter une charge ponctuelle", value=True)
if use_point_load:
    px = st.sidebar.slider("Position x charge ponctuelle [m]", min_value=0.0, max_value=float(L),
                            value=min(L / 4, L), step=0.05)
    py = st.sidebar.slider("Position y charge ponctuelle [m]", min_value=0.0, max_value=float(L),
                            value=min(L / 2, L), step=0.05)
    P_kN = st.sidebar.slider("Valeur P [kN]", min_value=0.0, max_value=1000.0,
                              value=550.0, step=10.0)
    pl_text = f"{px:.3f}, {py:.3f}, {P_kN:.3f}"
else:
    pl_text = ""

st.sidebar.header("Maillage")
n_mesh = st.sidebar.slider("Éléments par côté", min_value=8, max_value=60, value=32, step=2)

st.sidebar.header("Appuis")
sp_text = st.sidebar.text_area(
    "Points d'appui (x, y) — un par ligne",
    value=f"0.0, 0.0\n{L}, 0.0\n{L}, {L}\n0.0, {L}")

with st.sidebar.expander("Balayage charge / service"):
    q_sweep_text = st.text_input("Balayage q [kPa], séparé par des virgules",
                                  value="5, 10, 15, 20, 25, 30")
    DEFLECTION_LIMIT_DENOM = st.number_input("Limite flèche L/N", value=360, step=10)

with st.sidebar.expander("Paramètres d'armature"):
    cover = st.number_input("Enrobage [mm]", value=30.0, step=5.0) / 1000.0
    bar_diameter = st.number_input("Diamètre barre estimé [mm]", value=15.0, step=1.0) / 1000.0
    fy_CSA = st.number_input("fy CSA [MPa]", value=400.0, step=25.0) * 1e6
    phi_s_CSA = st.number_input("φs CSA A23.3", value=0.85, step=0.01)
    fy_ACI = st.number_input("fy ACI [MPa]", value=420.0, step=25.0) * 1e6
    phi_ACI = st.number_input("φ ACI 318", value=0.90, step=0.01)


def parse_triples(text):
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            out.append((float(parts[0]), float(parts[1]), float(parts[2]) * 1e3))
    return out


def parse_pairs(text):
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            out.append((float(parts[0]), float(parts[1])))
    return out


try:
    POINT_LOADS = parse_triples(pl_text)
    SUPPORT_POINTS = parse_pairs(sp_text)
    q_sweep = [float(v.strip()) * 1e3 for v in q_sweep_text.split(",") if v.strip()]
except ValueError:
    st.error("Format invalide dans les charges ponctuelles, appuis ou balayage — vérifier les virgules.")
    st.stop()

if len(SUPPORT_POINTS) < 3:
    st.error("Au moins 3 points d'appui non alignés sont requis.")
    st.stop()

# ============================================================================
# SOLVE ------------------------------------------------------------------
# ============================================================================

B = E * t**3 / (12.0 * (1 - nu**2))

mesh = PlateMesh(L, L, n_mesh, n_mesh)
supported, supported_xy = mesh.nearest_nodes(SUPPORT_POINTS)

pt_loads_used = []
for (xl, yl, P) in POINT_LOADS:
    node = mesh.nearest_node(xl, yl)
    xu, yu = mesh.nodes[node]
    pt_loads_used.append((xu, yu, P))

w, U = solve_plate_acm(mesh, B, nu, q, supported, point_loads=pt_loads_used if pt_loads_used else None)

center_node = mesh.node_id[n_mesh // 2, n_mesh // 2]
Wc = w[center_node]
Wmax_node = np.argmax(np.abs(w))
Wmax = w[Wmax_node]

X = mesh.nodes[:, 0].reshape(n_mesh + 1, n_mesh + 1)
Y = mesh.nodes[:, 1].reshape(n_mesh + 1, n_mesh + 1)
W = w.reshape(n_mesh + 1, n_mesh + 1) * 1000

sx = [p[0] for p in supported_xy]
sy = [p[1] for p in supported_xy]

mxx, myy, mxy, tx, ty = recover_moments(mesh, U, B, nu)
M1, M2, alpha = principal_moments(mxx, myy, mxy)

mxx_g = mxx.reshape(n_mesh + 1, n_mesh + 1)
myy_g = myy.reshape(n_mesh + 1, n_mesh + 1)
mxy_g = mxy.reshape(n_mesh + 1, n_mesh + 1)
tx_g = tx.reshape(n_mesh + 1, n_mesh + 1)
ty_g = ty.reshape(n_mesh + 1, n_mesh + 1)
M1_g = M1.reshape(n_mesh + 1, n_mesh + 1)
M2_g = M2.reshape(n_mesh + 1, n_mesh + 1)
alpha_g = alpha.reshape(n_mesh + 1, n_mesh + 1)

d_eff = t - cover - bar_diameter / 2.0
As_min_CSA = 0.002 * 1.0 * t
As_min_ACI = 0.0018 * 1.0 * t

As_CSA_xx = As_required(mxx, fy_CSA, phi_s_CSA, As_min_CSA, d_eff)
As_CSA_yy = As_required(myy, fy_CSA, phi_s_CSA, As_min_CSA, d_eff)
As_ACI_xx = As_required(mxx, fy_ACI, phi_ACI, As_min_ACI, d_eff)
As_ACI_yy = As_required(myy, fy_ACI, phi_ACI, As_min_ACI, d_eff)

As_CSA_xx_g = As_CSA_xx.reshape(n_mesh + 1, n_mesh + 1)
As_CSA_yy_g = As_CSA_yy.reshape(n_mesh + 1, n_mesh + 1)
As_ACI_xx_g = As_ACI_xx.reshape(n_mesh + 1, n_mesh + 1)
As_ACI_yy_g = As_ACI_yy.reshape(n_mesh + 1, n_mesh + 1)

d_mm = d_eff * 1000.0
b_mm = 1000.0

skip = max(1, n_mesh // 10)
Xq = X[::skip, ::skip]
Yq = Y[::skip, ::skip]
Aq = alpha_g[::skip, ::skip]
M1q = M1_g[::skip, ::skip]
M2q = M2_g[::skip, ::skip]
mxx_q = mxx_g[::skip, ::skip]
myy_q = myy_g[::skip, ::skip]
mxy_q = mxy_g[::skip, ::skip]

spacing = skip * (L / n_mesh)
bar_half_len = 0.40 * spacing

# rho1/rho2 computed on the SAME (subsampled) grid as M1q/M2q, so shapes
# match when draw_principal_bars masks rhofull by Mq (matches original script).
As1_bar = As_required(M1q, fy_CSA, phi_s_CSA, As_min_CSA, d_eff)
As2_bar = As_required(M2q, fy_CSA, phi_s_CSA, As_min_CSA, d_eff)
rho1 = As1_bar / (b_mm * d_mm)
rho2 = As2_bar / (b_mm * d_mm)

lw_min, lw_max = 1.0, 7.0


def rho_to_lw(rho):
    lo, hi = rho.min(), rho.max()
    if hi - lo < 1e-12:
        return np.full_like(rho, lw_max)
    return lw_min + (rho - lo) / (hi - lo) * (lw_max - lw_min)


lw1 = rho_to_lw(rho1)
lw2 = rho_to_lw(rho2)

mx_bot_g = np.maximum(mxx_g + np.abs(mxy_g), 0.0)
my_bot_g = np.maximum(myy_g + np.abs(mxy_g), 0.0)
mx_top_g = np.maximum(-(mxx_g - np.abs(mxy_g)), 0.0)
my_top_g = np.maximum(-(myy_g - np.abs(mxy_g)), 0.0)

mx_bot_q = mx_bot_g[::skip, ::skip]
my_bot_q = my_bot_g[::skip, ::skip]
mx_top_q = mx_top_g[::skip, ::skip]
my_top_q = my_top_g[::skip, ::skip]

RHO_CMAP = plt.cm.YlOrRd


def bar_linewidths(mag_q):
    As_q = As_required(mag_q, fy_CSA, phi_s_CSA, As_min_CSA, d_eff)
    rho_q = As_q / (b_mm * d_mm)
    return rho_to_lw(rho_q), rho_q


def draw_orthogonal_bars(ax, mx_q, my_q, linestyle):
    lw_x, rho_x = bar_linewidths(mx_q)
    lw_y, rho_y = bar_linewidths(my_q)
    vmin = min(rho_x.min(), rho_y.min())
    vmax = max(rho_x.max(), rho_y.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(Xq.shape[0]):
        for j in range(Xq.shape[1]):
            x0, y0 = Xq[i, j], Yq[i, j]
            if mx_q[i, j] > 0:
                ax.plot([x0 - bar_half_len, x0 + bar_half_len], [y0, y0],
                        color=RHO_CMAP(norm(rho_x[i, j])), linestyle=linestyle,
                        linewidth=lw_x[i, j], solid_capstyle='round')
            if my_q[i, j] > 0:
                ax.plot([x0, x0], [y0 - bar_half_len, y0 + bar_half_len],
                        color=RHO_CMAP(norm(rho_y[i, j])), linestyle=linestyle,
                        linewidth=lw_y[i, j], solid_capstyle='round')
    return rho_x, rho_y, norm


def draw_principal_bars(ax, sign_wanted, linestyle):
    drawn_rho = []
    for Mq, rhofull in [(M1q, rho1), (M2q, rho2)]:
        mask = (Mq >= 0) if sign_wanted > 0 else (Mq < 0)
        drawn_rho.append(rhofull[mask])
    drawn_rho = np.concatenate(drawn_rho) if drawn_rho and any(a.size for a in drawn_rho) else np.array([0.0])
    vmin, vmax = drawn_rho.min(), drawn_rho.max()
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))

    for Mq, lwfull, rhofull in [(M1q, lw1, rho1), (M2q, lw2, rho2)]:
        base_angle = Aq if Mq is M1q else (Aq + np.pi / 2)
        for i in range(Xq.shape[0]):
            for j in range(Xq.shape[1]):
                Mval = Mq[i, j]
                if (sign_wanted > 0 and Mval < 0) or (sign_wanted < 0 and Mval >= 0):
                    continue
                x0, y0 = Xq[i, j], Yq[i, j]
                ang = base_angle[i, j]
                dx, dy = np.cos(ang) * bar_half_len, np.sin(ang) * bar_half_len
                ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy],
                        color=RHO_CMAP(norm(rhofull[i, j])), linestyle=linestyle,
                        linewidth=lwfull[i, j], solid_capstyle='round')
    return norm


def base_ax(ax, title):
    ax.set_facecolor('#f7f2ea')
    ax.plot(sx, sy, 'ko', ms=8, zorder=5)
    if pt_loads_used:
        ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used],
                'g^', ms=11, zorder=5)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(title, fontsize=10)


def add_rho_colorbar(fig, ax, norm, face_label, linestyle):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=RHO_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85)
    cb.set_label('rho = As/(b.d)  [%]')
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{v*100:.2f}' for v in ticks])
    ax.legend(handles=[Line2D([0], [0], color='dimgray', lw=3, linestyle=linestyle,
                               label=face_label)],
              loc='upper right', fontsize=8, framealpha=0.9)


# ============================================================================
# UI: HEADER + KPIs -----------------------------------------------------------
# ============================================================================

st.title("Dalle 2D — Analyse et design (GCI2011)")

parts = []
if q != 0:
    parts.append(f"q={q/1e3:.1f} kPa")
if pt_loads_used:
    parts.append(", ".join(f"P={P/1e3:.0f}kN@({xu:.1f},{yu:.1f})" for (xu, yu, P) in pt_loads_used))
load_label = " + ".join(parts) if parts else "aucune charge"

N_worst = L / max(abs(Wmax), 1e-9)
status = "OK" if N_worst >= DEFLECTION_LIMIT_DENOM else "DÉPASSE LA LIMITE"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flèche au centre", f"{Wc*1000:.2f} mm")
c2.metric("Flèche max", f"{Wmax*1000:.2f} mm")
c3.metric("f/portée pire cas", f"L/{N_worst:.0f}", status)
c4.metric("Rigidité flexionnelle B", f"{B:.3e} N·m")
st.caption(f"Cas de charge : {load_label}")

tabs = st.tabs(["Flèche", "Moments & efforts", "Moments principaux",
                 "Armature — As", "Armature — plans"])

# ---- TAB 1: deflection -----------------------------------------------------
with tabs[0]:
    fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
    cf = ax1.contourf(X, Y, W, levels=20, cmap='viridis')
    cs = ax1.contour(X, Y, W, levels=10, colors='white', linewidths=0.5, alpha=0.6)
    ax1.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
    ax1.plot(sx, sy, 'ko', ms=7, label='appuis')
    if pt_loads_used:
        ax1.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used],
                  'r^', ms=10, label='charge ponctuelle')
        ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.set_title(f'Flèche W(x,y) [mm]  —  {load_label}')
    ax1.set_aspect('equal')
    fig1.colorbar(cf, label='W [mm]')
    plt.tight_layout()
    st.pyplot(fig1)

    col_a, col_b = st.columns(2)
    with col_a:
        Wc_list = []
        for qi in q_sweep:
            wi, _ = solve_plate_acm(mesh, B, nu, qi, supported,
                                     point_loads=pt_loads_used if pt_loads_used else None)
            Wc_list.append(wi[center_node] * 1000)
        x_axis = np.array(q_sweep) / 1e3
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.plot(x_axis, Wc_list, 'o-', color='#1f6f5c')
        ax2.set_xlabel('Charge uniforme q [kPa]')
        ax2.set_ylabel('Flèche au centre Wc [mm]')
        ax2.set_title('Flèche au centre vs. charge')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_b:
        eps = 1e-9
        N_ratio = L / np.maximum(np.abs(W / 1000.0), eps)
        N_plot = np.clip(N_ratio, 0, 6 * DEFLECTION_LIMIT_DENOM)
        fig3, ax3 = plt.subplots(figsize=(6.5, 5.5))
        cf3 = ax3.contourf(X, Y, N_plot, levels=25, cmap='RdYlGn')
        cs3 = ax3.contour(X, Y, N_ratio, levels=[DEFLECTION_LIMIT_DENOM], colors='black', linewidths=2)
        ax3.clabel(cs3, inline=True, fontsize=8, fmt=f'L/{DEFLECTION_LIMIT_DENOM} limite')
        ax3.plot(sx, sy, 'ko', ms=7)
        if pt_loads_used:
            ax3.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'r^', ms=10)
        ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]')
        ax3.set_title(f'f/portée = L/W  (ligne noire = limite L/{DEFLECTION_LIMIT_DENOM})')
        ax3.set_aspect('equal')
        fig3.colorbar(cf3, label='N (portée/N = flèche)')
        plt.tight_layout()
        st.pyplot(fig3)

# ---- TAB 2: moments & shears ------------------------------------------------
with tabs[1]:
    fig4, axs4 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, field, name in zip(axs4, [mxx_g, myy_g, mxy_g], ['mxx', 'myy', 'mxy']):
        cf = ax.contourf(X, Y, field / 1e3, levels=20, cmap='RdBu_r')
        ax.contour(X, Y, field / 1e3, levels=10, colors='k', linewidths=0.3, alpha=0.5)
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name}  [kN·m/m]'); ax.set_xlabel('x [m]')
        fig4.colorbar(cf, ax=ax, shrink=0.8)
    axs4[0].set_ylabel('y [m]')
    plt.tight_layout()
    st.pyplot(fig4)

    fig5, axs5 = plt.subplots(1, 2, figsize=(11, 5))
    for ax, field, name in zip(axs5, [tx_g, ty_g], ['tx', 'ty']):
        cf = ax.contourf(X, Y, field / 1e3, levels=20, cmap='PuOr')
        ax.contour(X, Y, field / 1e3, levels=10, colors='k', linewidths=0.3, alpha=0.5)
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name}  [kN/m]'); ax.set_xlabel('x [m]')
        fig5.colorbar(cf, ax=ax, shrink=0.8)
    axs5[0].set_ylabel('y [m]')
    plt.tight_layout()
    st.pyplot(fig5)

# ---- TAB 3: principal moments ----------------------------------------------
with tabs[2]:
    fig6, axs6 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, name in zip(axs6, [M1_g, M2_g], ['M1 (majeur)', 'M2 (mineur)']):
        cf = ax.contourf(X, Y, field / 1e3, levels=20, cmap='RdBu_r')
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name}  [kN·m/m]'); ax.set_xlabel('x [m]')
        fig6.colorbar(cf, ax=ax, shrink=0.8)
    axs6[0].set_ylabel('y [m]')
    plt.tight_layout()
    st.pyplot(fig6)

    max_arrow_len = 0.90 * spacing
    min_arrow_len = 0.15 * spacing

    def stretch(mag):
        lo, hi = mag.min(), mag.max()
        if hi - lo < 1e-12:
            return np.full_like(mag, max_arrow_len)
        return min_arrow_len + (mag - lo) / (hi - lo) * (max_arrow_len - min_arrow_len)

    len1 = stretch(np.abs(M1q))
    len2 = stretch(np.abs(M2q))
    Uq1, Vq1 = np.cos(Aq) * len1, np.sin(Aq) * len1
    Uq2, Vq2 = -np.sin(Aq) * len2, np.cos(Aq) * len2

    fig7, ax7 = plt.subplots(figsize=(6.5, 5.8))
    cf7 = ax7.contourf(X, Y, M1_g / 1e3, levels=20, cmap='Greys', alpha=0.5)
    ax7.quiver(Xq, Yq, Uq1, Vq1, color='crimson', pivot='mid',
               angles='xy', scale_units='xy', scale=1, width=0.008, label='M1 (longueur ~ |M1|)')
    ax7.quiver(Xq, Yq, -Uq1, -Vq1, color='crimson', pivot='mid',
               angles='xy', scale_units='xy', scale=1, width=0.008)
    ax7.quiver(Xq, Yq, Uq2, Vq2, color='navy', pivot='mid',
               angles='xy', scale_units='xy', scale=1, width=0.008, label='M2 (longueur ~ |M2|)')
    ax7.quiver(Xq, Yq, -Uq2, -Vq2, color='navy', pivot='mid',
               angles='xy', scale_units='xy', scale=1, width=0.008)
    ax7.plot(sx, sy, 'ko', ms=7)
    if pt_loads_used:
        ax7.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=10)
    ax7.set_aspect('equal'); ax7.set_xlabel('x [m]'); ax7.set_ylabel('y [m]')
    ax7.set_title('Trajectoires des moments principaux (longueur ~ magnitude)')
    ax7.legend(loc='upper right', fontsize=8)
    fig7.colorbar(cf7, ax=ax7, shrink=0.8, label='M1 [kN·m/m] (fond)')
    plt.tight_layout()
    st.pyplot(fig7)

# ---- TAB 4: As maps ----------------------------------------------------------
with tabs[3]:
    st.caption(f"d effectif = {d_eff*1000:.1f} mm — simplification As = M/(0.9·φ·fy·d), "
               "sans facteurs de charge. Voir code pour les limites de cette approche.")
    fig8, axs8 = plt.subplots(2, 2, figsize=(12, 10))
    panels = [(As_CSA_xx_g, 'CSA A23.3 — As, dir. x'), (As_CSA_yy_g, 'CSA A23.3 — As, dir. y'),
              (As_ACI_xx_g, 'ACI 318 — As, dir. x'), (As_ACI_yy_g, 'ACI 318 — As, dir. y')]
    for ax, (field, name) in zip(axs8.ravel(), panels):
        cf = ax.contourf(X, Y, field, levels=20, cmap='YlOrRd')
        ax.contour(X, Y, field, levels=10, colors='k', linewidths=0.3, alpha=0.4)
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'b^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name}  [mm²/m]')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        fig8.colorbar(cf, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig8)
    if pt_loads_used:
        st.info("As sous une charge ponctuelle hérite de la singularité de moment liée au maillage — "
                "ne pas dimensionner sur le pic brut; utiliser un moment moyenné sur la largeur de "
                "bande de colonne ou un périmètre de poinçonnement effectif.")

# ---- TAB 5: rebar layout sketches -------------------------------------------
with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        fig9a, ax9a = plt.subplots(figsize=(6.5, 5.8))
        norm9a = draw_principal_bars(ax9a, +1, '-')
        base_ax(ax9a, 'Système 1 (principal M1/M2) — nappe INFÉRIEURE\n(M > 0, flexion positive)')
        add_rho_colorbar(fig9a, ax9a, norm9a, 'Nappe inf. (trait plein)', '-')
        plt.tight_layout()
        st.pyplot(fig9a)

        fig10a, ax10a = plt.subplots(figsize=(6.5, 5.8))
        _, _, norm10a = draw_orthogonal_bars(ax10a, mx_bot_q, my_bot_q, '-')
        base_ax(ax10a, 'Système 2 (orthogonal, Wood-Armer) — nappe INFÉRIEURE\nMx+=mxx+|mxy|, My+=myy+|mxy|')
        add_rho_colorbar(fig10a, ax10a, norm10a, 'Nappe inf. (trait plein)', '-')
        plt.tight_layout()
        st.pyplot(fig10a)

    with col2:
        fig9b, ax9b = plt.subplots(figsize=(6.5, 5.8))
        norm9b = draw_principal_bars(ax9b, -1, (0, (4, 2)))
        base_ax(ax9b, 'Système 1 (principal M1/M2) — nappe SUPÉRIEURE\n(M < 0, flexion négative)')
        add_rho_colorbar(fig9b, ax9b, norm9b, 'Nappe sup. (tirets)', (0, (4, 2)))
        plt.tight_layout()
        st.pyplot(fig9b)

        fig10b, ax10b = plt.subplots(figsize=(6.5, 5.8))
        _, _, norm10b = draw_orthogonal_bars(ax10b, mx_top_q, my_top_q, (0, (4, 2)))
        base_ax(ax10b, 'Système 2 (orthogonal, Wood-Armer) — nappe SUPÉRIEURE\nMx-=|mxx-|mxy||, My-=|myy-|mxy||')
        add_rho_colorbar(fig10b, ax10b, norm10b, 'Nappe sup. (tirets)', (0, (4, 2)))
        plt.tight_layout()
        st.pyplot(fig10b)

    st.caption("Système 1 (principal) utilise moins d'acier total quand M1/M2 sont très différents "
               "en magnitude, mais nécessite des barres coupées et placées selon un angle variable — "
               "peu pratique pour la plupart des dalles coulées en place. Système 2 (orthogonal/"
               "Wood-Armer) est ce qui est réellement construit presque toujours.")
