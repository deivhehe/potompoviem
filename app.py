import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve, brentq
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


def _pick_physical_root(eqs, guesses, L_lid_mm, H_lid_mm, margin=0.25, reject_radius=2.0):
    x_lo, x_hi = -margin * L_lid_mm, (1 + margin) * L_lid_mm
    y_lo, y_hi = -margin * H_lid_mm, (1 + margin) * H_lid_mm

    candidates = []
    for g0 in guesses:
        sol, info, ier, msg = fsolve(eqs, g0, full_output=True)
        res = np.linalg.norm(info["fvec"])
        if res > 1e-3:
            continue
        if np.hypot(*sol) < reject_radius:
            continue
        key = (round(sol[0], 1), round(sol[1], 1))
        if any(key == c[2] for c in candidates):
            continue
        in_env = (x_lo <= sol[0] <= x_hi) and (y_lo <= sol[1] <= y_hi)
        candidates.append((res, sol, key, in_env))

    if not candidates:
        return None, np.inf, False

    in_env_candidates = [c for c in candidates if c[3]]
    if in_env_candidates:
        best = min(in_env_candidates, key=lambda c: c[0])
        return best[1], best[0], True

    best = min(candidates, key=lambda c: c[0])
    return best[1], best[0], False


def solve_main_pin_mm(Xb_mm, Yb_mm, L0_mm, S_mm, theta_max, L_lid_mm, H_lid_mm, min_x_cg=570.0):
    def eqs(v):
        lx, ly = v
        e1 = (lx - Xb_mm) ** 2 + (ly - Yb_mm) ** 2 - L0_mm ** 2
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        e2 = (Xp2 - Xb_mm) ** 2 + (Yp2 - Yb_mm) ** 2 - (L0_mm + S_mm) ** 2
        return [e1, e2]

    # Upravené odhady: cílít solver do oblasti X > min_x_cg (za těžištěm)
    guesses = [
        (min_x_cg + 50.0, H_lid_mm * 0.5),
        (min_x_cg + 100.0, H_lid_mm * 0.2),
        (min_x_cg, H_lid_mm * 0.8),
        (L_lid_mm * 0.7, H_lid_mm * 0.5),
        (L_lid_mm * 0.8, H_lid_mm * 0.3)
    ]
    
    res_sol, res_res, valid = _pick_physical_root(eqs, guesses, L_lid_mm, H_lid_mm)
    # Pokud by solver přesto našel něco před těžištěm, zkusíme to ohlídat
    return res_sol, res_res, valid


def solve_aux_pin_mm(Xb2_mm, Yb2_mm, L02_mm, theta_dead, L_lid_mm, H_lid_mm):
    R2 = Xb2_mm ** 2 + Yb2_mm ** 2
    R = np.sqrt(R2)
    if R < 1e-6:
        return None, np.inf, False, None

    sin_d, cos_d = np.sin(theta_dead), np.cos(theta_dead)
    disc = (L02_mm ** 2) / R2 - sin_d ** 2
    if disc < 0:
        return None, np.inf, False, R * abs(sin_d)

    sq = np.sqrt(disc)
    ux, uy = rotate_mm(Xb2_mm, Yb2_mm, -theta_dead)

    x_lo, x_hi = -0.15 * L_lid_mm, 1.15 * L_lid_mm
    y_lo, y_hi = -0.15 * H_lid_mm, 1.15 * H_lid_mm

    sols = []
    for t in (cos_d + sq, cos_d - sq):
        lx, ly = t * ux, t * uy
        if np.hypot(lx, ly) < 2.0:
            continue
        sols.append((lx, ly))

    if not sols:
        return None, np.inf, False, R * abs(sin_d)

    in_env = [s for s in sols if x_lo <= s[0] <= x_hi and y_lo <= s[1] <= y_hi]
    if in_env:
        return np.array(in_env[0]), 0.0, True, R * abs(sin_d)
    return np.array(sols[0]), 0.0, False, R * abs(sin_d)


def find_dead_point(cg_x_mm, cg_y_mm, theta_max):
    f = lambda th: cg_x_mm * np.cos(th) - cg_y_mm * np.sin(th)
    if f(0.0) * f(theta_max) >= 0:
        return None
    return brentq(f, 1e-6, theta_max)


# ----------------------------------------------------------------------
# UI - Sidebar
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
config = st.sidebar.radio("Typ", ["2× hlavní vzpěra", "2× hlavní + 2× pomocná vzpěra"])
use_aux = config.startswith("2× hlavní +")

st.sidebar.subheader("Hlavní vzpěra (1 ks)")
Xb1 = st.sidebar.number_input("Vana X (mm)", -1000.0, 3000.0, 585.0, 5.0)
Yb1 = st.sidebar.number_input("Vana Y (mm)", -1000.0, 1000.0, -111.0, 5.0)
L0_1 = st.sidebar.number_input("Zasunutá délka @0° (mm)", 30.0, 2000.0, 618.0, 5.0)
S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 500.0, 5.0)

# Minimální X pozice čepu na víku (zda má být za těžištěm, např. > 570 mm)
min_x_pin = st.sidebar.slider("Vynutit čep hlavní vzpěry za X (mm)", 0.0, lid_length, 570.0, 10.0)

if use_aux:
    st.sidebar.subheader("Pomocná (zadní) vzpěra (1 ks)")
    Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0)
    Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0)
    L0_2 = st.sidebar.number_input("Zasunutá délka pomocné @0° (mm)", 30.0, 2000.0, 561.0, 5.0)
    S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm) [info]", 10.0, 1500.0, 100.0, 5.0)
    F_aux_catalog = st.sidebar.number_input("Katalogová síla 1 ks pomocné vzpěry (N)", 50.0, 2000.0, 500.0, 10.0)

st.sidebar.header("6) Cílové síly do ruky")
target_open_N_input = st.sidebar.slider("Síla na otevření @0° (N)", 5.0, 150.0, 50.0, 5.0)

st.sidebar.header("7) Náhled úhlu")
theta_disp_deg = st.sidebar.slider("Úhel pro geometrický náhled (°)", 0, theta_max_deg, 0)
animate = st.sidebar.button("▶️ Animovat otevírání")

# ----------------------------------------------------------------------
# Výpočty
# ----------------------------------------------------------------------
theta_max = np.radians(theta_max_deg)
n_main = 2

def Xcg_m(theta):
    cg_xm = cg_x_mm * 0.001
    cg_ym = cg_y_mm * 0.001
    return cg_xm * np.cos(theta) - cg_ym * np.sin(theta)

def Tg(theta):
    return -lid_mass * G * Xcg_m(theta)

def handle_moment_arm_m(theta):
    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    r = np.hypot(hx, hy)
    return r * 0.001 if r > 1e-6 else 1e-6

pin1, res1, in_env1 = solve_main_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height, min_x_pin)

if pin1 is None:
    st.error("⚠️ Pro zadanou pozici čepu za touto X souřadnicí nelze najít platné geometrické řešení. Zkuste posunout limit X nebo zasunutou délku.")
    st.stop()

lx1, ly1 = pin1
theta_dead = find_dead_point(cg_x_mm, cg_y_mm, theta_max)

pin2 = None
n_aux = 2
if use_aux:
    if theta_dead is not None:
        pin2, res2, in_env2, min_L02 = solve_aux_pin_mm(Xb2, Yb2, L0_2, theta_dead, lid_length, lid_height)
        if pin2 is not None:
            lx2, ly2 = pin2

d1_0 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, 0.0)
h_arm_0 = handle_moment_arm_m(0.0)

if use_aux and pin2 is not None:
    d2_0 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, 0.0)
    moment_sum_needed = -(Tg(0.0) + target_open_N_input * h_arm_0)
    moment_from_aux = n_aux * d2_0 * F_aux_catalog
    denom = n_main * d1_0
    F_main = (moment_sum_needed - moment_from_aux) / denom if abs(denom) > 1e-9 else 0.0
    F_aux = F_aux_catalog
else:
    F_aux = None
    denom = n_main * d1_0
    F_main = (-(Tg(0.0) + target_open_N_input * h_arm_0)) / denom if abs(denom) > 1e-9 else 0.0

def F_hand(theta):
    Ts_main = n_main * F_main * signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta)
    Ts_aux = 0.0
    if use_aux and pin2 is not None and F_aux is not None:
        Ts_aux = n_aux * F_aux * signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, theta)
    
    h_arm = handle_moment_arm_m(theta)
    return -(Tg(theta) + Ts_main + Ts_aux) / h_arm

# ----------------------------------------------------------------------
# Metrický panel
# ----------------------------------------------------------------------
st.title("🔧 Návrh plynových vzpěr (S čepem za těžištěm)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Síla hlavní vzpěry (1 ks)", f"{F_main:.0f} N")
if use_aux and F_aux is not None:
    c2.metric("Síla pomocné vzpěry (1 ks)", f"{F_aux:.0f} N")
else:
    c2.metric("Síla pomocné vzpěry", "—")
c3.metric("Síla do ruky @0°", f"{F_hand(0.0):.1f} N")

f_max_val = F_hand(theta_max)
c4.metric(
    "Síla do ruky @max",
    f"{f_max_val:.1f} N",
    "pomáhá zavírat" if f_max_val < 0 else "tlačí ven",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Čep na víku – hlavní X", f"{lx1:.1f} mm")
c6.metric("Čep na víku – hlavní Y", f"{ly1:.1f} mm")
if use_aux and pin2 is not None:
    c7.metric("Čep na víku – pomocná X", f"{lx2:.1f} mm")
    c8.metric("Čep na víku – pomocná Y", f"{ly2:.1f} mm")
else:
    c7.metric("Čep na víku – pomocná X", "—")
    c8.metric("Čep na víku – pomocná Y", "—")

if theta_dead is not None:
    st.info(f"🔹 Mrtvý bod (těžiště nad pantem) při úhlu **{np.degrees(theta_dead):.1f}°**")
else:
    st.info("🔹 Mrtvý bod nebyl v zadaném rozsahu úhlů nalezen.")

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

    if use_aux and pin2 is not None:
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
