# -*- coding: utf-8 -*-
"""
Streamlit app: point/corner-supported rectangular slab, free edges,
uniform + point load(s) - Kirchhoff thin-plate FE (ACM 12-DOF element).

Run locally:   streamlit run app.py
Deploy: push this file + requirements.txt to a GitHub repo, then
        connect the repo at https://share.streamlit.io
"""
import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Slab FE — point supports, free edges", layout="wide")


# ============================================================================
# CORE FE CODE (mesh, ACM element, solver, moment recovery)
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


@st.cache_resource(show_spinner="Deriving ACM plate element (symbolic, once per server)...")
def build_acm_element():
    """ACM 12-DOF non-conforming thin plate element (no shear DOFs ->
    immune to shear locking / shear hourglassing). Returns Ke(a,b,D,nu), Fe(a,b,q)."""
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
    Bx, By, Bxy = dP(2, 0) * Cinv, dP(0, 2) * Cinv, 2 * dP(1, 1) * Cinv
    Bmat = sp.Matrix.vstack(Bx, By, Bxy)
    Db = D_s * sp.Matrix([[1, nu_s, 0], [nu_s, 1, 0], [0, 0, (1 - nu_s) / 2]])

    Ke_sym = (Bmat.T * Db * Bmat).applyfunc(
        lambda e: sp.integrate(sp.integrate(e, (x, 0, a)), (y, 0, b)))
    Fe_sym = (N.T * q_s).applyfunc(
        lambda e: sp.integrate(sp.integrate(e, (x, 0, a)), (y, 0, b)))

    Ke_func = sp.lambdify((a, b, D_s, nu_s), Ke_sym, 'numpy')
    Fe_func = sp.lambdify((a, b, q_s), Fe_sym, 'numpy')
    return Ke_func, Fe_func


@st.cache_resource(show_spinner="Deriving moment-recovery operators (symbolic, once per server)...")
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


def solve_plate_acm(mesh, Ke_func, Fe_func, D, nu, q, supported_nodes, point_loads=None):
    """DOF per node: (w, dw/dx, dw/dy). supported_nodes: w=0 prescribed there."""
    ex, ey = mesh.Lx / mesh.nx, mesh.Ly / mesh.ny
    Ke = np.array(Ke_func(ex, ey, D, nu), dtype=float)
    Fe = np.array(Fe_func(ex, ey, q), dtype=float).flatten()

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


def recover_moments(mesh, U, D, nu, mom_ops):
    ex, ey = mesh.Lx / mesh.nx, mesh.Ly / mesh.ny
    n_nodes = mesh.n_nodes
    corner_local = [(0, 0), (ex, 0), (ex, ey), (0, ey)]
    sums = {k: np.zeros(n_nodes) for k in ['mxx', 'myy', 'mxy', 'tx', 'ty']}
    counts = np.zeros(n_nodes)

    for el in mesh.elems:
        edofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in el]).ravel()
        d_e = U[edofs]
        for local_i, node in enumerate(el):
            xl, yl = corner_local[local_i]
            Wxx = float(np.array(mom_ops['xx'](xl, yl, ex, ey)).flatten() @ d_e)
            Wyy = float(np.array(mom_ops['yy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxy = float(np.array(mom_ops['xy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxxx = float(np.array(mom_ops['xxx'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxyy = float(np.array(mom_ops['xyy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wyyy = float(np.array(mom_ops['yyy'](xl, yl, ex, ey)).flatten() @ d_e)
            Wxxy = float(np.array(mom_ops['xxy'](xl, yl, ex, ey)).flatten() @ d_e)

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
    return avg + R, avg - R, 0.5 * np.arctan2(2 * mxy, (mxx - myy))


# ============================================================================
# SIDEBAR — INPUT
# ============================================================================

st.sidebar.header("Geometry & material")
L = st.sidebar.number_input("Slab side length L [m]", 0.5, 50.0, 4.0, 0.1)
t = st.sidebar.number_input("Thickness t [m]", 0.02, 2.0, 0.20, 0.01)
E_GPa = st.sidebar.number_input("Young's modulus E [GPa]", 1.0, 500.0, 25.0, 1.0)
nu = st.sidebar.slider("Poisson's ratio ν", 0.0, 0.49, 0.20, 0.01)
n_mesh = st.sidebar.slider("Mesh density (elements/side)", 8, 64, 32, 4)
E = E_GPa * 1e9

st.sidebar.header("Supports (w = 0, snapped to mesh)")
n_supports = st.sidebar.number_input("Number of supports", 3, 8, 4, 1)
default_corners = [(0.0, 0.0), (L, 0.0), (L, L), (0.0, L)]
support_points = []
for i in range(int(n_supports)):
    dx, dy = default_corners[i] if i < 4 else (L / 2, L / 2)
    c1, c2 = st.sidebar.columns(2)
    x_i = c1.number_input(f"support {i+1} x", 0.0, L, float(dx), 0.1, key=f"sx{i}")
    y_i = c2.number_input(f"support {i+1} y", 0.0, L, float(dy), 0.1, key=f"sy{i}")
    support_points.append((x_i, y_i))

st.sidebar.header("Loads")
q_kPa = st.sidebar.number_input("Uniform load q [kPa]", 0.0, 200.0, 10.0, 0.5)
q = q_kPa * 1e3

n_point_loads = st.sidebar.number_input("Number of point loads", 0, 6, 1, 1)
point_loads_input = []
for i in range(int(n_point_loads)):
    c1, c2, c3 = st.sidebar.columns(3)
    xp = c1.number_input(f"P{i+1} x", 0.0, L, float(L / 2), 0.1, key=f"px{i}")
    yp = c2.number_input(f"P{i+1} y", 0.0, L, float(L / 2), 0.1, key=f"py{i}")
    Pp = c3.number_input(f"P{i+1} [kN]", 0.0, 5000.0, 50.0, 5.0, key=f"pp{i}") * 1e3
    point_loads_input.append((xp, yp, Pp))

st.sidebar.header("Serviceability")
defl_limit_denom = st.sidebar.number_input("f/span limit denominator (L/N)", 100, 1000, 360, 10)

# ============================================================================
# SOLVE
# ============================================================================

B = E * t**3 / (12.0 * (1 - nu**2))
Ke_func, Fe_func = build_acm_element()
mom_ops = build_moment_recovery_ops()

mesh = PlateMesh(L, L, int(n_mesh), int(n_mesh))
supported, supported_xy = mesh.nearest_nodes(support_points)

pt_loads_used = []
for (xp, yp, P) in point_loads_input:
    if P > 0:
        node = mesh.nearest_node(xp, yp)
        pt_loads_used.append((*mesh.nodes[node], P))

w, U = solve_plate_acm(mesh, Ke_func, Fe_func, B, nu, q, supported,
                        point_loads=pt_loads_used if pt_loads_used else None)

center_node = mesh.node_id[int(n_mesh) // 2, int(n_mesh) // 2]
Wc = w[center_node]
Wmax_node = np.argmax(np.abs(w))
Wmax = w[Wmax_node]

mxx, myy, mxy, tx, ty = recover_moments(mesh, U, B, nu, mom_ops)
M1, M2, alpha = principal_moments(mxx, myy, mxy)

X = mesh.nodes[:, 0].reshape(int(n_mesh) + 1, int(n_mesh) + 1)
Y = mesh.nodes[:, 1].reshape(int(n_mesh) + 1, int(n_mesh) + 1)
sx = [p[0] for p in supported_xy]
sy = [p[1] for p in supported_xy]

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("Point/corner-supported slab — Kirchhoff plate FE")
st.caption("ACM 12-DOF non-conforming thin-plate element, validated against the classical "
           "simply-supported-square Navier benchmark (α = 0.00406, β = 0.0479 for ν = 0.3).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flexural rigidity B", f"{B:.3e} N·m")
c2.metric("Center deflection", f"{Wc*1000:.3f} mm")
c3.metric("Max deflection", f"{Wmax*1000:.3f} mm")
N_worst = L / max(abs(Wmax), 1e-9)
status = "OK" if N_worst >= defl_limit_denom else "EXCEEDS LIMIT"
c4.metric(f"Worst f/span (limit L/{defl_limit_denom})", f"L/{N_worst:.0f}", status)

tabs = st.tabs(["Deflection map", "Deflection vs. load", "f/span map",
                 "Moments mxx/myy/mxy", "Shears tx/ty",
                 "Principal moments M1/M2", "Principal directions"])

with tabs[0]:
    W_mm = w.reshape(int(n_mesh) + 1, int(n_mesh) + 1) * 1000
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.contourf(X, Y, W_mm, levels=20, cmap='viridis')
    ax.contour(X, Y, W_mm, levels=10, colors='white', linewidths=0.5, alpha=0.6)
    ax.plot(sx, sy, 'ko', ms=7, label='supports')
    if pt_loads_used:
        ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used],
                 'r^', ms=10, label='point load')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Deflection map W(x,y) [mm]')
    fig.colorbar(cf, label='W [mm]')
    st.pyplot(fig)

with tabs[1]:
    q_sweep_kPa = np.linspace(0, max(q_kPa, 10) * 1.5, 8)
    Wc_list = []
    for qi_kPa in q_sweep_kPa:
        wi, _ = solve_plate_acm(mesh, Ke_func, Fe_func, B, nu, qi_kPa * 1e3, supported,
                                 point_loads=pt_loads_used if pt_loads_used else None)
        Wc_list.append(wi[center_node] * 1000)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(q_sweep_kPa, Wc_list, 'o-', color='#1f6f5c')
    ax.axvline(q_kPa, color='crimson', linestyle='--', linewidth=1, label='current q')
    ax.set_xlabel('Uniform load q [kPa]  (point loads held fixed)')
    ax.set_ylabel('Center deflection Wc [mm]')
    ax.set_title('Center deflection vs. uniform load')
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tabs[2]:
    eps = 1e-9
    N_ratio = L / np.maximum(np.abs(w.reshape(int(n_mesh) + 1, int(n_mesh) + 1)), eps)
    N_plot = np.clip(N_ratio, 0, 6 * defl_limit_denom)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.contourf(X, Y, N_plot, levels=25, cmap='RdYlGn')
    cs = ax.contour(X, Y, N_ratio, levels=[defl_limit_denom], colors='black', linewidths=2)
    ax.clabel(cs, inline=True, fontsize=8, fmt=f'L/{defl_limit_denom} limit')
    ax.plot(sx, sy, 'ko', ms=7)
    if pt_loads_used:
        ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'r^', ms=10)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'f/span = L/W map (black line = L/{defl_limit_denom} limit)')
    fig.colorbar(cf, label='N (span/N = deflection, capped for display)')
    st.pyplot(fig)
    st.write(f"Worst-case f/span = **L/{N_worst:.0f}** (limit L/{defl_limit_denom}) -> **{status}**")

with tabs[3]:
    mxx_g = mxx.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    myy_g = myy.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    mxy_g = mxy.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, field, name in zip(axs, [mxx_g, myy_g, mxy_g], ['mxx', 'myy', 'mxy']):
        cf = ax.contourf(X, Y, field / 1e3, levels=20, cmap='RdBu_r')
        ax.contour(X, Y, field / 1e3, levels=10, colors='k', linewidths=0.3, alpha=0.5)
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name} [kN.m/m]'); ax.set_xlabel('x [m]')
        fig.colorbar(cf, ax=ax, shrink=0.8)
    axs[0].set_ylabel('y [m]')
    st.pyplot(fig)
    if pt_loads_used:
        st.info("Peak moment right under a point load is mesh-dependent (theoretically "
                "singular in Kirchhoff theory) — refining the mesh will keep raising it. "
                "Values away from the load converge normally.")

with tabs[4]:
    tx_g = tx.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    ty_g = ty.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, field, name in zip(axs, [tx_g, ty_g], ['tx', 'ty']):
        cf = ax.contourf(X, Y, field / 1e3, levels=20, cmap='PuOr')
        ax.contour(X, Y, field / 1e3, levels=10, colors='k', linewidths=0.3, alpha=0.5)
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name} [kN/m]'); ax.set_xlabel('x [m]')
        fig.colorbar(cf, ax=ax, shrink=0.8)
    axs[0].set_ylabel('y [m]')
    st.pyplot(fig)

with tabs[5]:
    M1_g = M1.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    M2_g = M2.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, name in zip(axs, [M1_g, M2_g], ['M1 (major)', 'M2 (minor)']):
        cf = ax.contourf(X, Y, field / 1e3, levels=20, cmap='RdBu_r')
        ax.plot(sx, sy, 'ko', ms=6)
        if pt_loads_used:
            ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=9)
        ax.set_aspect('equal'); ax.set_title(f'{name} [kN.m/m]'); ax.set_xlabel('x [m]')
        fig.colorbar(cf, ax=ax, shrink=0.8)
    axs[0].set_ylabel('y [m]')
    st.pyplot(fig)

with tabs[6]:
    alpha_g = alpha.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    M1_g = M1.reshape(int(n_mesh) + 1, int(n_mesh) + 1)
    M2_g = M2.reshape(int(n_mesh) + 1, int(n_mesh) + 1)

    skip = max(1, int(n_mesh) // 10)
    Xq, Yq = X[::skip, ::skip], Y[::skip, ::skip]
    Aq = alpha_g[::skip, ::skip]
    M1q, M2q = M1_g[::skip, ::skip], M2_g[::skip, ::skip]

    spacing = skip * (L / int(n_mesh))
    max_arrow_len = 0.90 * spacing
    min_arrow_len = 0.15 * spacing

    def stretch(mag):
        lo, hi = mag.min(), mag.max()
        if hi - lo < 1e-12:
            return np.full_like(mag, max_arrow_len)
        return min_arrow_len + (mag - lo) / (hi - lo) * (max_arrow_len - min_arrow_len)

    len1, len2 = stretch(np.abs(M1q)), stretch(np.abs(M2q))
    Uq1, Vq1 = np.cos(Aq) * len1, np.sin(Aq) * len1
    Uq2, Vq2 = -np.sin(Aq) * len2, np.cos(Aq) * len2

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    cf = ax.contourf(X, Y, M1_g / 1e3, levels=20, cmap='Greys', alpha=0.5)
    for sgn in (1, -1):
        ax.quiver(Xq, Yq, sgn * Uq1, sgn * Vq1, color='crimson', pivot='mid',
                   angles='xy', scale_units='xy', scale=1, width=0.008,
                   label='M1 (length ~ |M1|)' if sgn == 1 else None)
        ax.quiver(Xq, Yq, sgn * Uq2, sgn * Vq2, color='navy', pivot='mid',
                   angles='xy', scale_units='xy', scale=1, width=0.008,
                   label='M2 (length ~ |M2|)' if sgn == 1 else None)
    ax.plot(sx, sy, 'ko', ms=7)
    if pt_loads_used:
        ax.plot([p[0] for p in pt_loads_used], [p[1] for p in pt_loads_used], 'g^', ms=10)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Principal moment trajectories (arrow length ~ magnitude)')
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(cf, ax=ax, shrink=0.8, label='M1 [kN.m/m] (background)')
    st.pyplot(fig)
