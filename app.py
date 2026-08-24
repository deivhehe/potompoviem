import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq, minimize
import time

st.set_page_config(page_title="Návrh a kontrola plynových vzpěr víka", layout="wide")

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

def find_dead_point(cg_x_mm, cg_y_mm, theta_max):
    f = lambda th: cg_x_mm * np.cos(th) - cg_y_mm * np.sin(th)
    if f(0.0) * f(theta_max) >= 0:
        return None
    try:
        return brentq(f, 1e-6, theta_max)
    except ValueError:
        return None

def solve_main_pin_mm(Xb_mm, Yb_mm, L0_mm, S_mm, theta_max, L_lid_mm, H_lid_mm):
    def objective(v):
        lx, ly = v
        L_0 = np.sqrt((lx - Xb_mm)**2 + (ly - Yb_mm)**2)
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        L_max = np.sqrt((Xp2 - Xb_mm)**2 + (Yp2 - Yb_mm)**2)
        return (L_0 - L0_mm)**2 + (L_max - (L0_mm + S_mm))**2

    guesses = [[L_lid_mm * 0.5, H_lid_mm * 0.5], [L_lid_mm * 0.8, H_lid_mm * 0.2], [L_lid_mm * 0.2, H_lid_mm * 0.8]]
    for g0 in guesses:
        res = minimize(objective, g0, bounds=[(0.0, L_lid_mm), (0.0, H_lid_mm)], method='L-BFGS-B')
        if res.success and res.fun < 5.0:
            return res.x, True
    return None, False

def solve_pin_custom(Xb_mm, Yb_mm, L_min_mm, S_mm, theta_max, L_lid_mm, H_lid_mm, allow_behind=False):
    def obj(v):
        lx, ly = v
        L_0 = np.sqrt((lx - Xb_mm) ** 2 + (ly - Yb_mm) ** 2)
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        L_max_angle = np.sqrt((Xp2 - Xb_mm) ** 2 + (Yp2 - Yb_mm) ** 2)
        pen = 0.0
        if L_0 < L_min_mm or L_0 > L_min_mm + S_mm:
            pen += (min(abs(L_0 - L_min_mm), abs(L_0 - (L_min_mm + S_mm))))**2 * 10.0
        if L_max_angle < L_min_mm or L_max_angle > L_min_mm + S_mm:
            pen += (min(abs(L_max_angle - L_min_mm), abs(L_max_angle - (L_min_mm + S_mm))))**2 * 10.0
        return (L_0 - L_min_mm)**2 + pen

    x_bounds = (-400.0 if allow_behind else 0.0, L_lid_mm)
    res = minimize(obj, [100.0, 100.0], bounds=[x_bounds, (-500, H_lid_mm + 500)], method='L-BFGS-B')
    if res.success and res.fun < 100.0:
        return res.x, True
    return None, False

# ----------------------------------------------------------------------
# UI - Sidebar (Zadávané hodnoty)
# ----------------------------------------------------------------------
app_mode = st.sidebar.radio("Režim aplikace", ["Návrh a optimalizace (poloautomat)", "Kontrola existujícího řešení (manuál)", "Plný automat (Syntéza všeho)"])
st.sidebar.divider()

st.sidebar.header("1) Geometrie a hmotnost víka")
lid_length = st.sidebar.number_input("Délka víka (mm)", 50.0, 3000.0, 1109.0, 10.0)
lid_height = st.sidebar.number_input("Výška / tloušťka víka (mm)", 10.0, 1000.0, 812.0, 5.0)
lid_mass = st.sidebar.number_input("Hmotnost víka (kg)", 0.1, 500.0, 180.0, 0.5)

st.sidebar.header("2) Pozice madla (v zavřeném stavu)")
handle_x_mm = st.sidebar.number_input("Madlo X (mm)", 0.0, 3000.0, lid_length, 5.0)
handle_y_mm = st.sidebar.number_input("Madlo Y (mm)", -500.0, 1000.0, float(lid_height * 0.5), 5.0)

st.sidebar.header("3) Těžiště víka (v zavřeném stavu)")
cg_x_mm = st.sidebar.number_input("Těžiště X (mm)", 0.0, 3000.0, float(np.clip(lid_length * 0.5, 0.0, 3000.0)), 5.0)
cg_y_mm = st.sidebar.number_input("Těžiště Y (mm)", -500.0, 1000.0, float(np.clip(lid_height * 0.5, -500.0, 1000.0)), 5.0)

st.sidebar.header("4) Rozsah otevření")
theta_max_deg = st.sidebar.slider("Maximální úhel otevření (°)", 45, 130, 83)

st.sidebar.header("5) Konfigurace")
config_type = st.sidebar.radio("Typ uspořádání", ["2× hlavní vzpěra", "2× hlavní + 2× pomocná vzpěra"])
use_aux = (config_type == "2× hlavní + 2× pomocná vzpěra")

if app_mode == "Plný automat (Syntéza všeho)":
    st.sidebar.header("🎯 Cílové síly na madlu")
    target_f_open = st.sidebar.number_input("Cílová síla na otevření @0° (N)", 10.0, 500.0, 70.0)
    target_f_close = st.sidebar.number_input("Cílová síla na zavření @max (N)", -500.0, 0.0, -100.0)
elif use_aux:
    force_ratio = st.sidebar.slider("Podíl síly hlavní vzpěry (%)", 10, 90, 50, 5)

st.sidebar.header("6) Parametry vzpěr")
if app_mode != "Plný automat (Syntéza všeho)":
    Xb1 = st.sidebar.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0)
    Yb1 = st.sidebar.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0)

if app_mode in ["Návrh a optimalizace (poloautomat)", "Plný automat (Syntéza všeho)"]:
    L0_1 = st.sidebar.number_input("Zasunutá délka hlavní @0° (mm)", 30.0, 2000.0, 618.0, 5.0)
    S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 500.0, 5.0)
else:
    lx1_in = st.sidebar.number_input("Čep na víku X hlavní (mm)", -500.0, 3000.0, 342.0, 5.0)
    ly1_in = st.sidebar.number_input("Čep na víku Y hlavní (mm)", -500.0, 1000.0, 457.0, 5.0)

if use_aux:
    st.sidebar.subheader("Pomocná vzpěra")
    if app_mode != "Plný automat (Syntéza všeho)":
        Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0)
        Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0)
    if app_mode in ["Návrh a optimalizace (poloautomat)", "Plný automat (Syntéza všeho)"]:
        L_min_2 = st.sidebar.number_input("Min. zasunutá délka pomocné vzpěry (mm)", 30.0, 2000.0, 500.0, 5.0)
        S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm)", 10.0, 1500.0, 100.0, 5.0)
    else:
        lx2_in = st.sidebar.number_input("Čep na víku X pomocná (mm)", -500.0, 3000.0, 260.0, 5.0)
        ly2_in = st.sidebar.number_input("Čep na víku Y pomocná (mm)", -500.0, 1000.0, 308.0, 5.0)

st.sidebar.header("Náhled")
theta_disp_deg = st.sidebar.slider("Úhel pro geometrický náhled (°)", 0, theta_max_deg, 0)
animate = st.sidebar.button("▶️ Animovat otevírání")

# ----------------------------------------------------------------------
# Výpočetní jádro
# ----------------------------------------------------------------------
theta_max = np.radians(theta_max_deg)
n_main = 2
n_aux = 2
cg_xm, cg_ym = cg_x_mm * 0.001, cg_y_mm * 0.001

def Tg(theta):
    return -lid_mass * G * (cg_xm * np.cos(theta) - cg_ym * np.sin(theta))

def handle_moment_arm_m(theta):
    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    r = np.hypot(hx, hy)
    return r * 0.001 if r > 1e-6 else 1e-6

# ROZHODOVACÍ STROM REŽIMŮ
if app_mode == "Plný automat (Syntéza všeho)":
    st.warning("⚠️ Běží globální syntéza. Výsledky mohou být matematicky správné, ale konstrukčně nepraktické. Slouží k hrubému odhadu!")
    
    def global_objective(vars):
        if use_aux:
            Xb1, Yb1, lx1, ly1, Fm, Xb2, Yb2, lx2, ly2, Fa = vars
        else:
            Xb1, Yb1, lx1, ly1, Fm = vars
            Xb2 = Yb2 = lx2 = ly2 = Fa = 0.0
            
        pen = 0.0
        # Geo hlavní vzpěry
        L0_calc = np.hypot(lx1 - Xb1, ly1 - Yb1)
        Xp1, Yp1 = rotate_mm(lx1, ly1, theta_max)
        Lmax_calc = np.hypot(Xp1 - Xb1, Yp1 - Yb1)
        pen += (L0_calc - L0_1)**2 * 100
        pen += (Lmax_calc - (L0_1 + S1))**2 * 100
        
        # Geo pomocné vzpěry
        if use_aux:
            L0_a = np.hypot(lx2 - Xb2, ly2 - Yb2)
            Xp2, Yp2 = rotate_mm(lx2, ly2, theta_max)
            Lmax_a = np.hypot(Xp2 - Xb2, Yp2 - Yb2)
            if L0_a < L_min_2 or L0_a > L_min_2 + S2: pen += 10000
            if Lmax_a < L_min_2 or Lmax_a > L_min_2 + S2: pen += 10000
            
        # Momentová bilance
        h_arm0 = handle_moment_arm_m(0.0)
        h_arm_max = handle_moment_arm_m(theta_max)
        
        d1_0 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, 0.0)
        d2_0 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, 0.0) if use_aux else 0.0
        F_hand_0 = -(Tg(0.0) + n_main*Fm*d1_0 + n_aux*Fa*d2_0) / h_arm0
        
        d1_m = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta_max)
        d2_m = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, theta_max) if use_aux else 0.0
        F_hand_m = -(Tg(theta_max) + n_main*Fm*d1_m + n_aux*Fa*d2_m) / h_arm_max
        
        err = (F_hand_0 - target_f_open)**2 + (F_hand_m - target_f_close)**2
        return err + pen
        
    if use_aux:
        x0 = [lid_length*0.5, -100, lid_length*0.5, lid_height*0.5, 500.0, 100, -100, 100, 100, 500.0]
        bounds = [(-500, lid_length), (-500, lid_height), (0, lid_length), (0, lid_height), (10, 5000),
                  (-500, lid_length), (-500, lid_height), (-200, lid_length), (0, lid_height), (10, 5000)]
    else:
        x0 = [lid_length*0.5, -100, lid_length*0.5, lid_height*0.5, 500.0]
        bounds = [(-500, lid_length), (-500, lid_height), (0, lid_length), (0, lid_height), (10, 5000)]
        
    res_glob = minimize(global_objective, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 2000})
    if use_aux:
        Xb1, Yb1, lx1, ly1, F_main, Xb2, Yb2, lx2, ly2, F_aux = res_glob.x
    else:
        Xb1, Yb1, lx1, ly1, F_main = res_glob.x
        Xb2 = Yb2 = lx2 = ly2 = F_aux = 0.0

else:
    # ------------------ STÁVAJÍCÍ REŽIMY (Návrh / Kontrola) ------------------
    if app_mode == "Návrh a optimalizace (poloautomat)":
        pin1, ok1 = solve_main_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height)
        if not ok1: st.error("Nelze geometricky vyřešit čep hlavní vzpěry."); st.stop()
        lx1, ly1 = pin1
        if use_aux:
            pin2, ok2 = solve_pin_custom(Xb2, Yb2, L_min_2, S2, theta_max, lid_length, lid_height, allow_behind=True)
            if not ok2: st.error("Nelze geometricky vyřešit čep pomocné vzpěry."); st.stop()
            lx2, ly2 = pin2
        else:
            lx2, ly2 = 0.0, 0.0
    else:
        lx1, ly1 = lx1_in, ly1_in
        if use_aux: lx2, ly2 = lx2_in, ly2_in
        else: lx2, ly2 = 0.0, 0.0

    # Optimalizace samotných sil (jako předtím)
    def solve_forces():
        def objective(forces):
            if use_aux:
                Fm, Fa = forces
                if Fm < 10 or Fa < 10: return 1e9
                ratio_penalty = ( (Fm/(Fm+Fa+1e-6)) - (force_ratio/100.0) )**2 * 100000.0
            else:
                Fm, Fa, ratio_penalty = forces[0], 0.0, 0.0
                if Fm < 10: return 1e9
            err = 0.0
            for th in np.linspace(0, theta_max, 5):
                d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, th)
                d2 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, th) if use_aux else 0.0
                err += ((n_main*Fm*d1 + n_aux*Fa*d2) - (-Tg(th)))**2
            return err + ratio_penalty
            
        if use_aux:
            res = minimize(objective, [500.0, 500.0], bounds=[(10, 5000), (10, 5000)], method='L-BFGS-B')
            return res.x[0], res.x[1] if res.success else (500.0, 500.0)
        else:
            res = minimize(objective, [500.0], bounds=[(10, 5000)], method='L-BFGS-B')
            return res.x[0], 0.0 if res.success else (500.0, 0.0)

    F_main, F_aux = solve_forces()


def calc_F_hand_internal(th, Fm, Fa):
    Ts_main = n_main * Fm * signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, th)
    Ts_aux = n_aux * Fa * signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, th) if use_aux else 0.0
    return -(Tg(th) + Ts_main + Ts_aux) / handle_moment_arm_m(th)

def F_hand(theta): return calc_F_hand_internal(theta, F_main, F_aux)
theta_dead = find_dead_point(cg_x_mm, cg_y_mm, theta_max)

# ----------------------------------------------------------------------
# Metrický panel a Vykreslování
# ----------------------------------------------------------------------
st.title(f"🔧 {app_mode}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Potřebná síla hlavní (1 ks)", f"{F_main:.0f} N")
if use_aux: c2.metric("Potřebná síla pomocné (1 ks)", f"{F_aux:.0f} N")
else: c2.metric("Pomocná vzpěra", "—")
c3.metric("Síla na madlo k otevření @0°", f"{F_hand(0.0):.1f} N")
f_max_val = F_hand(theta_max)
c4.metric("Síla na madlo k zavření @max", f"{f_max_val:.1f} N", "Drží víko v otevřené pozici" if f_max_val < 0 else "tlačí ven")

c5, c6, c7, c8 = st.columns(4)
if app_mode == "Plný automat (Syntéza všeho)":
    c5.metric("Hledaná Vana / Čep hlavní X", f"{Xb1:.1f} / {lx1:.1f} mm")
    c6.metric("Hledaná Vana / Čep hlavní Y", f"{Yb1:.1f} / {ly1:.1f} mm")
    if use_aux:
        c7.metric("Hledaná Vana / Čep pom. X", f"{Xb2:.1f} / {lx2:.1f} mm")
        c8.metric("Hledaná Vana / Čep pom. Y", f"{Yb2:.1f} / {ly2:.1f} mm")
else:
    c5.metric("Čep na víku – hlavní X", f"{lx1:.1f} mm")
    c6.metric("Čep na víku – hlavní Y", f"{ly1:.1f} mm")
    if use_aux:
        c7.metric("Čep na víku – pomocná X", f"{lx2:.1f} mm")
        c8.metric("Čep na víku – pomocná Y", f"{ly2:.1f} mm")

if theta_dead is not None:
    st.info(f"🔹 Mrtvý bod při úhlu **{np.degrees(theta_dead):.1f}°**")

st.divider()

col_geo, col_force = st.columns(2)
fig1, ax1 = plt.subplots(figsize=(4, 4))
fig2, ax2 = plt.subplots(figsize=(4, 4))
common_adjust = {'left': 0.2, 'bottom': 0.2, 'right': 0.95, 'top': 0.9}
fig1.subplots_adjust(**common_adjust)
fig2.subplots_adjust(**common_adjust)

theta_disp = np.radians(theta_disp_deg)

def draw_geometry_mm(ax, theta):
    ax.clear()
    corners_local = [(0, 0), (lid_length, 0), (lid_length, lid_height), (0, lid_height)]
    corners_global = [rotate_mm(lx, ly, theta) for lx, ly in corners_local]
    xs = [p[0] for p in corners_global] + [corners_global[0][0]]
    ys = [p[1] for p in corners_global] + [corners_global[0][1]]
    ax.fill(xs, ys, color="#c9a876", alpha=0.6, edgecolor="black", linewidth=1.5, zorder=3)
    ax.plot(0, 0, "ko", markersize=8, zorder=5)
    Xc, Yc = rotate_mm(cg_x_mm, cg_y_mm, theta)
    ax.plot(Xc, Yc, "o", color="red", markersize=9, zorder=6)
    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    ax.plot(hx, hy, "go", markersize=9, zorder=7)
    
    Xp1, Yp1 = rotate_mm(lx1, ly1, theta)
    ax.plot([Xb1, Xp1], [Yb1, Yp1], "-", color="#1f77b4", linewidth=3, zorder=4)
    ax.plot(Xb1, Yb1, "s", color="#1f77b4", markersize=7, zorder=5)
    ax.plot(Xp1, Yp1, "^", color="#1f77b4", markersize=7, zorder=5)

    if use_aux:
        Xp2, Yp2 = rotate_mm(lx2, ly2, theta)
        ax.plot([Xb2, Xp2], [Yb2, Yp2], "-", color="#d62728", linewidth=3, zorder=4)
        ax.plot(Xb2, Yb2, "s", color="#d62728", markersize=7, zorder=5)
        ax.plot(Xp2, Yp2, "^", color="#d62728", markersize=7, zorder=5)

    max_dim = max(lid_length, lid_height)
    ax.set_xlim(-max_dim * 0.25, lid_length * 1.2)
    ax.set_ylim(-400, max(lid_height * 1.5, 300))
    ax.set_box_aspect(1)
    ax.invert_xaxis()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_title(f"Geometrie @ {np.degrees(theta):.1f}°", fontsize=10, fontweight='bold')
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
        ax.axvline(np.degrees(theta_dead), color="purple", linestyle="--", linewidth=1.5)

    if theta_marker is not None:
        ax.plot(np.degrees(theta_marker), F_hand(theta_marker), "o", color="black", markersize=8, zorder=6)

    ax.set_box_aspect(1)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlabel("Úhel otevření (°)", fontsize=9)
    ax.set_ylabel("Síla na madlu (N)", fontsize=9)
    ax.set_title("Profil síly na madlu", fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

if animate:
    p1, p2 = col_geo.empty(), col_force.empty()
    for deg in np.linspace(0, theta_max_deg, 40):
        th = np.radians(deg)
        draw_geometry_mm(ax1, th)
        draw_force_profile(ax2, th)
        p1.pyplot(fig1)
        p2.pyplot(fig2)
        time.sleep(0.04)
else:
    draw_geometry_mm(ax1, theta_disp)
    draw_force_profile(ax2, theta_disp)
    col_geo.pyplot(fig1)
    col_force.pyplot(fig2)
