import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve, brentq, minimize
import time

st.set_page_config(page_title="Návrh plynových vzpěr víka", layout="wide")

G = 9.81  # m/s^2

# ----------------------------------------------------------------------
# Pomocné fyzikální funkce
# ----------------------------------------------------------------------
def rotate_mm(lx, ly, theta):
    c, s = np.cos(theta), np.sin(theta)
    return lx * c - ly * s, lx * s + ly * c


def signed_moment_arm_mm(Xb_mm, Yb_mm, lx_mm, ly_mm, theta):
    Xp_mm, Yp_mm = rotate_mm(lx_mm, ly_mm, theta)
    L_mm = np.sqrt((Xp_mm - Xb_mm) ** 2 + (Yp_mm - Yb_mm) ** 2)
    if L_mm < 1e-6:
        return 0.0
    return (Xb_mm * Yp_mm - Yb_mm * Xp_mm) / (L_mm * 1000.0)


def solve_pin_mm(Xb_mm, Yb_mm, L0_mm, S_mm, theta_max, L_lid_mm, H_lid_mm):
    def eqs(v):
        lx, ly = v
        e1 = (lx - Xb_mm) ** 2 + (ly - Yb_mm) ** 2 - L0_mm ** 2
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        e2 = (Xp2 - Xb_mm) ** 2 + (Yp2 - Yb_mm) ** 2 - (L0_mm + S_mm) ** 2
        return [e1, e2]

    guesses = [
        (L_lid_mm * 0.5, H_lid_mm * 0.5),
        (L_lid_mm * 0.75, H_lid_mm * 0.3),
        (L_lid_mm * 0.3, H_lid_mm * 0.7),
        (L_lid_mm * 0.9, H_lid_mm * 0.2)
    ]
    
    for g0 in guesses:
        sol, info, ier, _ = fsolve(eqs, g0, full_output=True)
        if ier == 1 and np.linalg.norm(info["fvec"]) < 1e-3:
            return sol, True
    return None, False


def find_dead_point(cg_x_mm, cg_y_mm, theta_max):
    f = lambda th: cg_x_mm * np.cos(th) - cg_y_mm * np.sin(th)
    if f(0.0) * f(theta_max) >= 0:
        return None
    try:
        return brentq(f, 1e-6, theta_max)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# UI - Sidebar (Zadávané hodnoty)
# ----------------------------------------------------------------------
st.sidebar.header("1) Geometrie a hmotnost víka")
lid_length = st.sidebar.number_input("Délka víka (mm)", 50.0, 3000.0, 1109.0, 10.0)
lid_height = st.sidebar.number_input("Výška / tloušťka víka (mm)", 10.0, 1000.0, 812.0, 5.0)
lid_mass = st.sidebar.number_input("Hmotnost víka (kg)", 0.1, 500.0, 180.0, 0.5)

st.sidebar.header("2) Pozice madla (v zavřeném stavu, od pantu)")
handle_x_mm = st.sidebar.number_input("Madlo X (mm)", 0.0, 3000.0, lid_length, 5.0)
handle_y_mm = st.sidebar.number_input("Madlo Y (mm)", -500.0, 1000.0, float(lid_height * 0.5), 5.0)

st.sidebar.header("3) Těžiště víka (od pantu, v zavřeném stavu)")
cg_x_mm = st.sidebar.number_input("Těžiště X (mm)", 0.0, 3000.0, float(np.clip(lid_length * 0.5, 0.0, 3000.0)), 5.0)
cg_y_mm = st.sidebar.number_input("Těžiště Y (mm)", -500.0, 1000.0, float(np.clip(lid_height * 0.5, -500.0, 1000.0)), 5.0)

st.sidebar.header("4) Rozsah otevření")
theta_max_deg = st.sidebar.slider("Maximální úhel otevření (°)", 45, 130, 83)

st.sidebar.header("5) Konfigurace vzpěr")
config_type = st.sidebar.radio("Typ uspořádání", ["2× hlavní vzpěra", "2× hlavní + 2× pomocná vzpěra"])
use_aux = (config_type == "2× hlavní + 2× pomocná vzpěra")

st.sidebar.header("6) Hlavní vzpěra")
Xb1 = st.sidebar.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0)
Yb1 = st.sidebar.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0)
L0_1 = st.sidebar.number_input("Zasunutá délka hlavní @0° (mm)", 30.0, 2000.0, 618.0, 5.0)
S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 500.0, 5.0)

if use_aux:
    st.sidebar.header("7) Pomocná vzpěra")
    Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0)
    Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0)
    L0_2 = st.sidebar.number_input("Zasunutá délka pomocné @0° (mm)", 30.0, 2000.0, 561.0, 5.0)
    S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm)", 10.0, 1500.0, 100.0, 5.0)

st.sidebar.header("8) Náhled úhlu")
theta_disp_deg = st.sidebar.slider("Úhel pro geometrický náhled (°)", 0, theta_max_deg, 0)
animate = st.sidebar.button("▶️ Animovat otevírání")

# ----------------------------------------------------------------------
# Výpočetní jádro
# ----------------------------------------------------------------------
theta_max = np.radians(theta_max_deg)
n_main = 2
n_aux = 2

# Výpočet pozic čepů na víku
pin1, ok1 = solve_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height)
if not ok1:
    st.error("⚠️ Pro zadané parametry hlavní vzpěry nelze geometricky vyřešit pozici čepu.")
    st.stop()
lx1, ly1 = pin1

if use_aux:
    pin2, ok2 = solve_pin_mm(Xb2, Yb2, L0_2, S2, theta_max, lid_length, lid_height)
    if not ok2:
        st.error("⚠️ Pro zadané parametry pomocné vzpěry nelze geometricky vyřešit pozici čepu.")
        st.stop()
    lx2, ly2 = pin2
else:
    lx2, ly2 = 0.0, 0.0

theta_dead = find_dead_point(cg_x_mm, cg_y_mm, theta_max)

cg_xm, cg_ym = cg_x_mm * 0.001, cg_y_mm * 0.001
def Tg(theta):
    return -lid_mass * G * (cg_xm * np.cos(theta) - cg_ym * np.sin(theta))

def solve_forces():
    def objective(forces):
        if use_aux:
            Fm, Fa = forces
            if Fm < 10 or Fa < 10:
                return 1e9
        else:
            Fm = forces[0]
            if Fm < 10:
                return 1e9
            Fa = 0.0
        
        err = 0.0
        for th in np.linspace(0, theta_max, 5):
            d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, th)
            if use_aux:
                d2 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, th)
                moment_vzpěr = n_main * Fm * d1 + n_aux * Fa * d2
            else:
                moment_vzpěr = n_main * Fm * d1
            moment_tíže = -Tg(th)
            err += (moment_vzpěr - moment_tíže)**2
        return err

    if use_aux:
        res = minimize(objective, [500.0, 500.0], bounds=[(10, 5000), (10, 5000)], method='L-BFGS-B')
        if res.success:
            return res.x[0], res.x[1]
        return 500.0, 500.0
    else:
        res = minimize(objective, [500.0], bounds=[(10, 5000)], method='L-BFGS-B')
        if res.success:
            return res.x[0], 0.0
        return 500.0, 0.0

F_main, F_aux = solve_forces()

def handle_moment_arm_m(theta):
    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    r = np.hypot(hx, hy)
    return r * 0.001 if r > 1e-6 else 1e-6

def F_hand(theta):
    Ts_main = n_main * F_main * signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta)
    Ts_aux = n_aux * F_aux * signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, theta) if use_aux else 0.0
    h_arm = handle_moment_arm_m(theta)
    return -(Tg(theta) + Ts_main + Ts_aux) / h_arm

# ----------------------------------------------------------------------
# Metrický panel
# ----------------------------------------------------------------------
st.title("🔧 Návrh plynových vzpěr víka (Přímý výpočet geometrie)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Síla hlavní vzpěry (1 ks)", f"{F_main:.0f} N")
if use_aux:
    c2.metric("Síla pomocné vzpěry (1 ks)", f"{F_aux:.0f} N")
else:
    c2.metric("Síla pomocné vzpěry", "—")
c3.metric("Potřebná síla k otevření", f"{F_hand(0.0):.1f} N")
f_max_val = F_hand(theta_max)
c4.metric("Potřebná síla k zavření", f"{f_max_val:.1f} N", "Drží víko v otevřené pozici" if f_max_val < 0 else "tlačí ven")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Čep na víku – hlavní X", f"{lx1:.1f} mm")
c6.metric("Čep na víku – hlavní Y", f"{ly1:.1f} mm")
if use_aux:
    c7.metric("Čep na víku – pomocná X", f"{lx2:.1f} mm")
    c8.metric("Čep na víku – pomocná Y", f"{ly2:.1f} mm")
else:
    c7.metric("Čep na víku – pomocná X", "—")
    c8.metric("Čep na víku – pomocná Y", "—")

if theta_dead is not None:
    st.info(f"🔹 Mrtvý bod při úhlu **{np.degrees(theta_dead):.1f}°**")

st.divider()

# ----------------------------------------------------------------------
# Vykreslení geometrie a grafu
# ----------------------------------------------------------------------
def draw_geometry_mm(ax, theta):
    ax.clear()
    corners_local = [(0, 0), (lid_length, 0), (lid_length, lid_height), (0, lid_height)]
    corners_global = [rotate_mm(lx, ly, theta) for lx, ly in corners_local]
    xs = [p[0] for p in corners_global] + [corners_global[0][0]]
    ys = [p[1] for p in corners_global] + [corners_global[0][1]]
    ax.fill(xs, ys, color="#c9a876", alpha=0.6, edgecolor="black", linewidth=1.5, zorder=3)

    ax.plot(0, 0, "ko", markersize=8, zorder=5)
    ax.annotate("Pant", (0, 0), textcoords="offset points", xytext=(-8, -12), fontsize=9, fontweight='bold')

    Xc, Yc = rotate_mm(cg_x_mm, cg_y_mm, theta)
    ax.plot(Xc, Yc, "o", color="red", markersize=9, zorder=6)
    ax.annotate("CG", (Xc, Yc), textcoords="offset points", xytext=(6, 6), color="red", fontsize=9, fontweight='bold')

    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    ax.plot(hx, hy, "go", markersize=9, zorder=7)
    ax.annotate("Madlo", (hx, hy), textcoords="offset points", xytext=(6, 6), color="green", fontsize=9, fontweight='bold')

    Xp1, Yp1 = rotate_mm(lx1, ly1, theta)
    ax.plot([Xb1, Xp1], [Yb1, Yp1], "-", color="#1f77b4", linewidth=3, zorder=4, label="Hlavní vzpěra")
    ax.plot(Xb1, Yb1, "s", color="#1f77b4", markersize=7, zorder=5)
    ax.plot(Xp1, Yp1, "^", color="#1f77b4", markersize=7, zorder=5)

    if use_aux:
        Xp2, Yp2 = rotate_mm(lx2, ly2, theta)
        ax.plot([Xb2, Xp2], [Yb2, Yp2], "-", color="#d62728", linewidth=3, zorder=4, label="Pomocná vzpěra")
        ax.plot(Xb2, Yb2, "s", color="#d62728", markersize=7, zorder=5)
        ax.plot(Xp2, Yp2, "^", color="#d62728", markersize=7, zorder=5)

    max_dim = max(lid_length, lid_height)
    ax.set_xlim(-max_dim * 0.15, lid_length * 1.2)
    ax.set_ylim(-400, max(lid_height * 1.5, 300))
    ax.set_box_aspect(1)
    ax.invert_xaxis()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_title(f"Geometrie víka @ {np.degrees(theta):.1f}°", fontsize=10, fontweight='bold')
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)


def draw_force_profile(ax, theta_marker=None):
    ax.clear()
    thetas = np.linspace(0, theta_max, 200)
    forces_n = np.array([F_hand(t) for t in thetas])
    degs = np.degrees(thetas)

    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(degs, forces_n, 0, where=(forces_n >= 0), color="#ff7f0e", alpha=0.5, label="Nutno tlačit")
    ax.fill_between(degs, forces_n, 0, where=(forces_n < 0), color="#2ca02c", alpha=0.5, label="Vzpěra pomáhá")
    ax.plot(degs, forces_n, color="black", linewidth=1.5)

    if theta_dead is not None:
        ax.axvline(np.degrees(theta_dead), color="purple", linestyle="--", linewidth=1.5, label="Mrtvý bod")

    if theta_marker is not None:
        ax.plot(np.degrees(theta_marker), F_hand(theta_marker), "o", color="black", markersize=8, zorder=6)

    ax.set_box_aspect(1)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlabel("Úhel otevření (°)", fontsize=9)
    ax.set_ylabel("Síla na madlu (N)", fontsize=9)
    ax.set_title("Profil síly na madlu", fontsize=10, fontweight='bold')
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3)


col_geo, col_force = st.columns(2)
fig1, ax1 = plt.subplots(figsize=(4, 4))
fig2, ax2 = plt.subplots(figsize=(4, 4))
common_adjust = {'left': 0.2, 'bottom': 0.2, 'right': 0.95, 'top': 0.9}
fig1.subplots_adjust(**common_adjust)
fig2.subplots_adjust(**common_adjust)

theta_disp = np.radians(theta_disp_deg)

if animate:
    placeholder1 = col_geo.empty()
    placeholder2 = col_force.empty()
    for deg in np.linspace(0, theta_max_deg, 40):
        th = np.radians(deg)
        draw_geometry_mm(ax1, th)
        draw_force_profile(ax2, th)
        placeholder1.pyplot(fig1)
        placeholder2.pyplot(fig2)
        time.sleep(0.04)
else:
    draw_geometry_mm(ax1, theta_disp)
    draw_force_profile(ax2, theta_disp)
    col_geo.pyplot(fig1)
    col_force.pyplot(fig2)
