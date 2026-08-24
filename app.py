import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq, minimize
import time

st.set_page_config(page_title="Návrh a kontrola plynových vzpěr víka", layout="wide")

G = 9.81  # m/s^2

# Přesné typy vzpěr a jejich nárůst síly (progrese) dle tvého zadání
STRUT_TYPES = {
    "Typ G3/8 (nárůst 30 %)": 0.30,
    "Typ G4/12 (nárůst 25 %)": 0.25,
    "Typ G6/15 (nárůst 35 %)": 0.35,
    "Typ G8/19 (nárůst 35 %)": 0.35,
    "Typ G10/23 (nárůst 35 %)": 0.35,
    "Typ G10/14 (nárůst 50 %)": 0.50,
    "Typ G20/40 (nárůst 35 %)": 0.35,
    "Typ G22/40 (nárůst 45 %)": 0.45,
    "Typ G25/55 (nárůst 35 %)": 0.35,
    "Typ G30/65 (nárůst 35 %)": 0.35,
    "Vlastní koeficient nárůstu": 0.35
}

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

def solve_main_pin_mm(Xb_mm, Yb_mm, L0_mm, S_mm, theta_max, L_lid_mm, H_lid_mm):
    def objective(v):
        lx, ly = v
        L_0 = np.sqrt((lx - Xb_mm)**2 + (ly - Yb_mm)**2)
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        L_max = np.sqrt((Xp2 - Xb_mm)**2 + (Yp2 - Yb_mm)**2)
        return (L_0 - L0_mm)**2 + (L_max - (L0_mm + S_mm))**2

    guesses = [
        [L_lid_mm * 0.5, H_lid_mm * 0.5],
        [L_lid_mm * 0.8, H_lid_mm * 0.2],
        [L_lid_mm * 0.2, H_lid_mm * 0.8],
        [100.0, 100.0]
    ]
    
    for g0 in guesses:
        res = minimize(objective, g0, bounds=[(0.0, L_lid_mm), (0.0, H_lid_mm)], method='L-BFGS-B')
        if res.success and res.fun < 10.0:
            return res.x, True
            
    res_fallback = minimize(objective, [L_lid_mm * 0.5, H_lid_mm * 0.5], bounds=[(-100.0, L_lid_mm + 200), (-200.0, H_lid_mm + 200)], method='L-BFGS-B')
    if res_fallback.success:
        return res_fallback.x, True

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
    if res.success and res.fun < 500.0:
        return res.x, True
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
app_mode = st.sidebar.radio("Režim aplikace", ["Návrh a optimalizace", "Kontrola existujícího řešení"])
st.sidebar.divider()

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

st.sidebar.header("5) Konfigurace vzpěr a nárůst síly")
config_type = st.sidebar.radio("Typ uspořádání", ["2× hlavní vzpěra", "2× hlavní + 2× pomocná vzpěra"])
use_aux = (config_type == "2× hlavní + 2× pomocná vzpěra")

strut_type_key = st.sidebar.selectbox("Typ vzpěry (progresivita)", list(STRUT_TYPES.keys()))
progression_rate = STRUT_TYPES[strut_type_key]
if strut_type_key == "Vlastní koeficient nárůstu":
    progression_rate = st.sidebar.slider("Vlastní nárůst síly (%)", 0.0, 1.0, 0.35, 0.05)

# Přepínač pro vlastní sílu v režimu kontroly
use_custom_forces = False
if app_mode == "Kontrola existujícího řešení":
    st.sidebar.header("6) Síly vzpěr")
    use_custom_forces = st.sidebar.checkbox("Vlastní síla vzpěr (přebít výpočet)", value=False)
    if use_custom_forces:
        custom_f_main = st.sidebar.number_input("Jmenovitá síla F1 (1 ks hlavní)", 10.0, 10000.0, 500.0, 10.0)
        if use_aux:
            custom_f_aux = st.sidebar.number_input("Jmenovitá síla F1 (1 ks pomocné)", 10.0, 10000.0, 300.0, 10.0)

if not use_custom_forces:
    if use_aux:
        st.sidebar.header("6) Poměr sil vzpěr")
        force_ratio = st.sidebar.slider("Podíl síly hlavní vzpěry (%)", 10, 90, 50, 5)
    else:
        force_ratio = 50

st.sidebar.header("7) Hlavní vzpěra")
Xb1 = st.sidebar.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0)
Yb1 = st.sidebar.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0)

if app_mode == "Návrh a optimalizace":
    L0_1 = st.sidebar.number_input("Zasunutá délka hlavní @0° (mm)", 30.0, 2000.0, 618.0, 5.0)
    S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 500.0, 5.0)
else:
    lx1_in = st.sidebar.number_input("Čep na víku X hlavní (mm)", -500.0, 3000.0, 342.0, 5.0)
    ly1_in = st.sidebar.number_input("Čep na víku Y hlavní (mm)", -500.0, 1000.0, 457.0, 5.0)

if use_aux:
    st.sidebar.header("8) Pomocná vzpěra (konzolka)")
    Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0)
    Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0)
    if app_mode == "Návrh a optimalizace":
        L_min_2 = st.sidebar.number_input("Min. zasunutá délka pomocné vzpěry (mm)", 30.0, 2000.0, 500.0, 5.0)
        S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm)", 10.0, 1500.0, 100.0, 5.0)
    else:
        lx2_in = st.sidebar.number_input("Čep na víku X pomocná (mm)", -500.0, 3000.0, 260.0, 5.0)
        ly2_in = st.sidebar.number_input("Čep na víku Y pomocná (mm)", -500.0, 1000.0, 308.0, 5.0)

st.sidebar.header("9) Náhled")
theta_disp_deg = st.sidebar.slider("Úhel pro geometrický náhled (°)", 0, theta_max_deg, 0)
animate = st.sidebar.button("▶️ Animovat otevírání")

# ----------------------------------------------------------------------
# Výpočetní jádro (sdílené)
# ----------------------------------------------------------------------
theta_max = np.radians(theta_max_deg)
n_main = 2
n_aux = 2

if app_mode == "Návrh a optimalizace":
    pin1, ok1 = solve_main_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height)
    if not ok1:
        st.error("⚠️ Pro zadané parametry hlavní vzpěry nelze ideálně geometricky vyřešit pozici čepu. Zkuste upravit délku nebo zdvih vzpěry.")
        st.stop()
    lx1, ly1 = pin1

    if use_aux:
        pin2, ok2 = solve_pin_custom(Xb2, Yb2, L_min_2, S2, theta_max, lid_length, lid_height, allow_behind=True)
        if not ok2:
            st.error("⚠️ Pro zadané parametry pomocné vzpěry nelze najít pozici čepu.")
            st.stop()
        lx2, ly2 = pin2
    else:
        lx2, ly2 = 0.0, 0.0
else:
    lx1, ly1 = lx1_in, ly1_in
    if use_aux:
        lx2, ly2 = lx2_in, ly2_in
    else:
        lx2, ly2 = 0.0, 0.0

theta_dead = find_dead_point(cg_x_mm, cg_y_mm, theta_max)
cg_xm, cg_ym = cg_x_mm * 0.001, cg_y_mm * 0.001

def Tg(theta):
    return -lid_mass * G * (cg_xm * np.cos(theta) - cg_ym * np.sin(theta))

def handle_moment_arm_m(theta):
    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    r = np.hypot(hx, hy)
    return r * 0.001 if r > 1e-6 else 1e-6

def get_strut_force_at_length(L_current, L_min, S, F_nominal):
    compression_ratio = np.clip(( (L_min + S) - L_current ) / (S + 1e-6), 0.0, 1.0)
    return F_nominal * (1.0 + progression_rate * compression_ratio)

def get_struts_forces_at_angle(th, Fm_nom, Fa_nom):
    Xp1, Yp1 = rotate_mm(lx1, ly1, th)
    L1 = np.sqrt((Xp1 - Xb1)**2 + (Yp1 - Yb1)**2)
    L_min1 = np.sqrt((lx1 - Xb1)**2 + (ly1 - Yb1)**2)
    Xp1_m, Yp1_m = rotate_mm(lx1, ly1, theta_max)
    L_max1 = np.sqrt((Xp1_m - Xb1)**2 + (Yp1_m - Yb1)**2)
    S1_est = max(abs(L_max1 - L_min1), 10.0)
    Fm_actual = get_strut_force_at_length(L1, min(L_min1, L_max1), S1_est, Fm_nom)

    Fa_actual = 0.0
    if use_aux:
        Xp2, Yp2 = rotate_mm(lx2, ly2, th)
        L2 = np.sqrt((Xp2 - Xb2)**2 + (Yp2 - Yb2)**2)
        L_min2 = np.sqrt((lx2 - Xb2)**2 + (ly2 - Yb2)**2)
        Xp2_m, Yp2_m = rotate_mm(lx2, ly2, theta_max)
        L_max2 = np.sqrt((Xp2_m - Xb2)**2 + (Yp2_m - Yb2)**2)
        S2_est = max(abs(L_max2 - L_min2), 10.0)
        Fa_actual = get_strut_force_at_length(L2, min(L_min2, L_max2), S2_est, Fa_nom)

    return Fm_actual, Fa_actual

def solve_forces():
    if use_custom_forces:
        Fm = custom_f_main
        Fa = custom_f_aux if use_aux else 0.0
        return Fm, Fa

    def objective(forces):
        if use_aux:
            Fm_nom, Fa_nom = forces
            if Fm_nom < 10 or Fa_nom < 10:
                return 1e9
            ratio_actual = Fm_nom / (Fm_nom + Fa_nom + 1e-6)
            ratio_target = force_ratio / 100.0
            ratio_penalty = (ratio_actual - ratio_target)**2 * 100000.0
        else:
            Fm_nom = forces[0]
            if Fm_nom < 10:
                return 1e9
            Fa_nom = 0.0
            ratio_penalty = 0.0
        
        err = 0.0
        for th in np.linspace(0, theta_max, 5):
            d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, th)
            Fm_act, Fa_act = get_struts_forces_at_angle(th, Fm_nom, Fa_nom)
            if use_aux:
                d2 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, th)
                moment_vzpěr = n_main * Fm_act * d1 + n_aux * Fa_act * d2
            else:
                moment_vzpěr = n_main * Fm_act * d1
            moment_tíže = -Tg(th)
            err += (moment_vzpěr - moment_tíže)**2
        return err + ratio_penalty

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

def calc_F_hand_internal(th, Fm_nom, Fa_nom):
    d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, th)
    d2 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, th) if use_aux else 0.0
    Fm_act, Fa_act = get_struts_forces_at_angle(th, Fm_nom, Fa_nom)
    Ts_main = n_main * Fm_act * d1
    Ts_aux = n_aux * Fa_act * d2 if use_aux else 0.0
    h_arm = handle_moment_arm_m(th)
    return -(Tg(th) + Ts_main + Ts_aux) / h_arm

def F_hand(theta):
    return calc_F_hand_internal(theta, F_main, F_aux)

# ----------------------------------------------------------------------
# Metrický panel a Vykreslování
# ----------------------------------------------------------------------
st.title(f"🔧 {app_mode}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jmenovitá síla F1 hlavní (1 ks)", f"{F_main:.0f} N")
if use_aux:
    c2.metric("Jmenovitá síla F1 pomocné (1 ks)", f"{F_aux:.0f} N")
else:
    c2.metric("Pomocná vzpěra", "—")
c3.metric("Potřebná síla k otevření @0°", f"{F_hand(0.0):.1f} N")
f_max_val = F_hand(theta_max)
c4.metric("Potřebná síla k zavření @max", f"{f_max_val:.1f} N", "Drží víko v otevřené pozici" if f_max_val < 0 else "tlačí ven")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Čep na víku – hlavní X", f"{lx1:.1f} mm")
c6.metric("Čep na víku – hlavní Y", f"{ly1:.1f} mm")
if use_aux:
    c7.metric("Čep na víku – pomocná X", f"{lx2:.1f} mm", "mimo víko (konzolka)" if lx2 < 0 else "")
    c8.metric("Čep na víku – pomocná Y", f"{ly2:.1f} mm")
else:
    c7.metric("Čep na víku – pomocná X", "—")
    c8.metric("Čep na víku – pomocná Y", "—")

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
    ax.set_xlim(-max_dim * 0.25, lid_length * 1.2)
    ax.set_ylim(-400, max(lid_height * 1.5, 300))
    ax.set_box_aspect(1)
    ax.invert_xaxis()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_title(f"Geometrie @ {np.degrees(theta):.1f}°", fontsize=10, fontweight='bold')
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
