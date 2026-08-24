import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve, brentq, minimize
import time

st.set_page_config(page_title="Návrh a kontrola plynových vzpěr víka", layout="wide")

st.markdown("""
    <style>
    @media print {
        @page {
            size: A4 landscape;
            margin: 0mm;
        }
        header, footer, .stButton, .viewerBadge_container__1QSob, [data-testid="stSidebar"], [data-testid="stHeader"] {
            display: none !important;
        }
        hr {
            display: none !important;
        }
        [data-testid="stMain"] {
            margin-left: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
        }
        .block-container {
            max-width: 100% !important;
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        body {
            zoom: 50% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

G = 9.81  # m/s^2

STRUT_TYPES = {
    "Typ G3/8 (nárůst 30 %)": 0.30,
    "Typ G4/12 (nárůst 25 %)": 0.25,
    "Typ G6/15 (nárůst 35 %)": 0.35,
    "Typ G8/19 (nárůst 35 %)": 0.35,
    "Typ G10/23 (nárůst 35 %)": 0.35,
    "Typ G14/28 (nárůst 50 %)": 0.50,
    "Typ G20/40 (nárůst 35 %)": 0.35,
    "Typ G22/40 (nárůst 45 %)": 0.45,
    "Typ G25/55 (nárůst 35 %)": 0.35,
    "Typ G30/65 (nárůst 35 %)": 0.35,
    "Vlastní koeficient nárůstu": 0.35
}

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

    res = minimize(objective, [L_lid_mm * 0.5, H_lid_mm * 0.5], bounds=[(10.0, L_lid_mm), (10.0, H_lid_mm)], method='L-BFGS-B')
    if res.success:
        return res.x, True
    return [L_lid_mm * 0.5, H_lid_mm * 0.5], False

def solve_pin_custom(Xb_mm, Yb_mm, L_min_mm, S_mm, theta_max, L_lid_mm, H_lid_mm, allow_behind=False, n_theta=40):
    # OPRAVA: dřív se posuzovaly jen dva body dráhy (theta=0 a theta_max).
    # Vzdálenost rotujícího čepu od pevné vany ale obecně NENÍ monotónní funkce
    # úhlu - může se uprostřed dráhy nejdřív zmenšit a pak zase zvětšit (nebo naopak).
    # Proto se teď posuzuje CELÁ trajektorie: cílíme na to, aby L(0) = L_min a
    # aby MAXIMUM délky na celé dráze bylo u L_min+S (skutečné využití zdvihu),
    # a penalizujeme jakýkoli bod dráhy mimo fyzický rozsah [L_min, L_min+S].
    thetas = np.linspace(0.0, theta_max, n_theta)

    def obj(v):
        lx, ly = v
        L_0 = np.sqrt((lx - Xb_mm) ** 2 + (ly - Yb_mm) ** 2)

        Xp_all, Yp_all = rotate_mm(lx, ly, thetas)
        L_path = np.sqrt((Xp_all - Xb_mm) ** 2 + (Yp_all - Yb_mm) ** 2)

        err = (L_0 - L_min_mm) ** 2
        err += (L_path.max() - (L_min_mm + S_mm)) ** 2

        below = np.clip(L_min_mm - L_path, 0.0, None)
        above = np.clip(L_path - (L_min_mm + S_mm), 0.0, None)
        err += 20.0 * np.sum(below ** 2) + 20.0 * np.sum(above ** 2)
        return err

    x_bounds = (-400.0 if allow_behind else 10.0, L_lid_mm)
    y_bounds = (-200.0, H_lid_mm + 500.0)

    # Vícenásobný start, aby optimalizátor neuvízl ve špatném lokálním minimu
    # (typicky degenerované řešení s téměř nulovým skutečným zdvihem).
    start_points = [
        [100.0, 100.0],
        [L_lid_mm * 0.15, H_lid_mm * 0.15],
        [L_lid_mm * 0.35, H_lid_mm * 0.5],
        [L_lid_mm * 0.1, H_lid_mm * 0.75],
        [max(x_bounds[0], -200.0), H_lid_mm * 0.3],
    ]

    best_res = None
    for x0 in start_points:
        x0c = [float(np.clip(x0[0], x_bounds[0], x_bounds[1])), float(np.clip(x0[1], y_bounds[0], y_bounds[1]))]
        res = minimize(obj, x0c, bounds=[x_bounds, y_bounds], method='L-BFGS-B')
        if res.success and (best_res is None or res.fun < best_res.fun):
            best_res = res

    if best_res is not None and best_res.fun < 500.0:
        return best_res.x, True
    return [100.0, 100.0], False

def find_dead_point(cg_x_mm, cg_y_mm, theta_max):
    f = lambda th: cg_x_mm * np.cos(th) - cg_y_mm * np.sin(th)
    if f(0.0) * f(theta_max) >= 0:
        return None
    try:
        return brentq(f, 1e-6, theta_max)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# UI - Sidebar
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

strut_type_main_key = st.sidebar.selectbox("Typ hlavní vzpěry (progresivita)", list(STRUT_TYPES.keys()))
progression_rate_main = STRUT_TYPES[strut_type_main_key]
if strut_type_main_key == "Vlastní koeficient nárůstu":
    progression_rate_main = st.sidebar.slider("Vlastní nárůst hlavní vzpěry (%)", 0.0, 1.0, 0.35, 0.05)

if use_aux:
    strut_type_aux_key = st.sidebar.selectbox("Typ pomocné vzpěry (progresivita)", list(STRUT_TYPES.keys()))
    progression_rate_aux = STRUT_TYPES[strut_type_aux_key]
    if strut_type_aux_key == "Vlastní koeficient nárůstu":
        progression_rate_aux = st.sidebar.slider("Vlastní nárůst pomocné vzpěry (%)", 0.0, 1.0, 0.35, 0.05)
else:
    progression_rate_aux = 0.35

use_custom_forces = False
if app_mode == "Kontrola existujícího řešení":
    st.sidebar.header("6) Síly vzpěr")
    use_custom_forces = st.sidebar.checkbox("Vlastní síla vzpěr (přebít výpočet)", value=True)
    if use_custom_forces:
        custom_f_main = st.sidebar.number_input("Jmenovitá síla F1 (1 ks hlavní)", 10.0, 10000.0, 650.0, 10.0)
        if use_aux:
            custom_f_aux = st.sidebar.number_input("Jmenovitá síla F1 (1 ks pomocné)", 10.0, 10000.0, 325.0, 10.0)

st.sidebar.header("7) Hlavní vzpěra")
Xb1 = st.sidebar.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0)
Yb1 = st.sidebar.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0)

L0_1 = st.sidebar.number_input("Zasunutá délka hlavní @0° (mm)", 30.0, 2000.0, 618.0, 5.0)
S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 500.0, 5.0)

# Výchozí hodnoty pro pomocnou vzpěru (aby proměnné vždy existovaly, i když use_aux == False)
L2_min_input = 478.0
S2 = 120.0

if use_aux:
    st.sidebar.header("Pomocná vzpěra (konzolka)")
    Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0)
    Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0)
    L2_min_input = st.sidebar.number_input("Minimální délka pomocné vzpěry @0° (mm)", 100.0, 2000.0, 478.0, 1.0)
    S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm)", 10.0, 1500.0, 120.0, 5.0)

# Volitelná ruční úprava čepů
st.sidebar.header("8) Interaktivní úprava čepů")
enable_manual_pin = st.sidebar.checkbox("Ručně upravit pozice čepů na víku", value=False)

st.sidebar.header("9) Ovládání náhledu a animace")
if 'anim_deg' not in st.session_state:
    st.session_state.anim_deg = 0.0

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶ Play"):
    st.session_state.is_playing = True
if col_btn2.button("⏸ Pause"):
    st.session_state.is_playing = False

if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

theta_disp_deg = st.sidebar.slider("Úhel otevření (°)", 0.0, float(theta_max_deg), value=float(st.session_state.anim_deg), step=1.0)
if theta_disp_deg != st.session_state.anim_deg:
    st.session_state.anim_deg = theta_disp_deg

# ----------------------------------------------------------------------
# Výpočet výchozích optimálních čepů
# ----------------------------------------------------------------------
theta_max = np.radians(theta_max_deg)
n_main, n_aux = 2, 2
pin_def, ok_def = solve_main_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height)
default_lx1, default_ly1 = pin_def if ok_def else (lid_length * 0.5, lid_height * 0.5)

if use_aux:
    # OPRAVA: dřív se tu volalo natvrdo solve_pin_custom(Xb2, Yb2, 300.0, 120.0, ...),
    # takže se ignorovaly skutečné hodnoty L2_min_input / S2 zadané v sidebaru.
    pin2, ok2 = solve_pin_custom(Xb2, Yb2, L2_min_input, S2, theta_max, lid_length, lid_height, allow_behind=True)
    default_lx2, default_ly2 = pin2 if ok2 else (100.0, 100.0)
else:
    default_lx2, default_ly2 = 0.0, 0.0

theta_dead = find_dead_point(cg_x_mm, cg_y_mm, theta_max)
cg_xm, cg_ym = cg_x_mm * 0.001, cg_y_mm * 0.001

def Tg(theta):
    return -lid_mass * G * (cg_xm * np.cos(theta) - cg_ym * np.sin(theta))

def handle_moment_arm_m(theta):
    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    r = np.hypot(hx, hy)
    return r * 0.001 if r > 1e-6 else 1e-6

def get_strut_force_at_length(L_current, L_min, S, F_nominal, prog_rate):
    compression_ratio = np.clip(( (L_min + S) - L_current ) / (S + 1e-6), 0.0, 1.0)
    return F_nominal * (1.0 + prog_rate * compression_ratio)

def get_struts_forces_at_angle(th, Fm_nom, Fa_nom, cur_lx1, cur_ly1, cur_lx2, cur_ly2):
    Xp1, Yp1 = rotate_mm(cur_lx1, cur_ly1, th)
    L1 = np.sqrt((Xp1 - Xb1)**2 + (Yp1 - Yb1)**2)
    Fm_actual = get_strut_force_at_length(L1, L0_1, S1, Fm_nom, progression_rate_main)

    Fa_actual = 0.0
    if use_aux:
        # OPRAVA: dřív se tu používal "delta trik" - L2_min_input + (geom_th - geom_0) -
        # ten mlčky předpokládal, že geometrická vzdálenost čepů při theta=0 se rovná
        # zadané minimální délce vzpěry L2_min_input. V ručním režimu (a i v auto režimu,
        # pokud solver nekonverguje přesně) to ale neplatí a vznikají fyzikálně nesmyslné
        # hodnoty. Fyzická délka vzpěry mezi dvěma čepy je prostě jejich eukleidovská
        # vzdálenost - žádný přepočet přes theta=0 není potřeba.
        Xp2, Yp2 = rotate_mm(cur_lx2, cur_ly2, th)
        L2_current = np.sqrt((Xp2 - Xb2) ** 2 + (Yp2 - Yb2) ** 2)
        Fa_actual = get_strut_force_at_length(L2_current, L2_min_input, S2, Fa_nom, progression_rate_aux)

    return Fm_actual, Fa_actual

def aux_strut_travel_mm(cur_lx2, cur_ly2, th_max, n_points=200):
    """Vrátí (min, max) SKUTEČNÉ geometrické délky pomocné vzpěry (přímá vzdálenost
    čep-vana) na celé trajektorii 0..th_max, a úhly, ve kterých min/max nastává."""
    thetas = np.linspace(0.0, th_max, n_points)
    Xp_all, Yp_all = rotate_mm(cur_lx2, cur_ly2, thetas)
    lengths = np.sqrt((Xp_all - Xb2) ** 2 + (Yp_all - Yb2) ** 2)
    imin, imax = int(np.argmin(lengths)), int(np.argmax(lengths))
    return float(lengths.min()), float(lengths.max()), float(np.degrees(thetas[imin])), float(np.degrees(thetas[imax]))

if 'last_solved_forces' not in st.session_state:
    st.session_state.last_solved_forces = [500.0, 300.0]

def solve_forces_smooth(lx_opt, ly_opt, lx2_opt, ly2_opt, target_force_at_0=None):
    if use_custom_forces and app_mode == "Kontrola existujícího řešení":
        return custom_f_main, (custom_f_aux if use_aux else 0.0)

    initial_guess = st.session_state.last_solved_forces

    if use_aux:
        def objective(vars_forces):
            Fm_nom, Fa_nom = vars_forces
            if Fm_nom < 10 or Fa_nom < 10:
                return 1e12
            
            err = 0.0
            for th in np.linspace(0, theta_max, 5):
                d1 = signed_moment_arm_mm(Xb1, Yb1, lx_opt, ly_opt, th)
                d2 = signed_moment_arm_mm(Xb2, Yb2, lx2_opt, ly2_opt, th)
                Fm_act, Fa_act = get_struts_forces_at_angle(th, Fm_nom, Fa_nom, lx_opt, ly_opt, lx2_opt, ly2_opt)
                moment_vzpěr = n_main * Fm_act * d1 + n_aux * Fa_act * d2
                moment_tíže = -Tg(th)
                err += (moment_vzpěr - moment_tíže)**2
            
            if target_force_at_0 is not None:
                d1_0 = signed_moment_arm_mm(Xb1, Yb1, lx_opt, ly_opt, 0.0)
                d2_0 = signed_moment_arm_mm(Xb2, Yb2, lx2_opt, ly2_opt, 0.0)
                Fm_0, Fa_0 = get_struts_forces_at_angle(0.0, Fm_nom, Fa_nom, lx_opt, ly_opt, lx2_opt, ly2_opt)
                Ts_0 = n_main * Fm_0 * d1_0 + n_aux * Fa_0 * d2_0
                h_arm_0 = handle_moment_arm_m(0.0)
                current_f_0 = -(Tg(0.0) + Ts_0) / h_arm_0
                err += (current_f_0 - target_force_at_0)**2 * 50.0

            return err

        res = minimize(objective, initial_guess, bounds=[(10, 20000), (10, 20000)], method='L-BFGS-B')
        if res.success:
            st.session_state.last_solved_forces = [res.x[0], res.x[1]]
            return res.x[0], res.x[1]
        return initial_guess[0], initial_guess[1]
    else:
        def objective_single(Fm_nom):
            if Fm_nom[0] < 10:
                return 1e12
            err = 0.0
            for th in np.linspace(0, theta_max, 5):
                d1 = signed_moment_arm_mm(Xb1, Yb1, lx_opt, ly_opt, th)
                Fm_act, _ = get_struts_forces_at_angle(th, Fm_nom[0], 0.0, lx_opt, ly_opt, 0, 0)
                moment_vzpěr = n_main * Fm_act * d1
                moment_tíže = -Tg(th)
                err += (moment_vzpěr - moment_tíže)**2
            
            if target_force_at_0 is not None:
                d1_0 = signed_moment_arm_mm(Xb1, Yb1, lx_opt, ly_opt, 0.0)
                Fm_0, _ = get_struts_forces_at_angle(0.0, Fm_nom[0], 0.0, lx_opt, ly_opt, 0, 0)
                Ts_0 = n_main * Fm_0 * d1_0
                h_arm_0 = handle_moment_arm_m(0.0)
                current_f_0 = -(Tg(0.0) + Ts_0) / h_arm_0
                err += (current_f_0 - target_force_at_0)**2 * 50.0

            return err

        res = minimize(objective_single, [initial_guess[0]], bounds=[(10, 20000)], method='L-BFGS-B')
        if res.success:
            st.session_state.last_solved_forces = [res.x[0], 0.0]
            return res.x[0], 0.0
        return initial_guess[0], 0.0

# ----------------------------------------------------------------------
# Určení pozic čepů a korekce počáteční síly
# ----------------------------------------------------------------------
force_offset_at_0 = 0.0

# OPRAVA: dřív bylo podmíněno i app_mode == "Návrh a optimalizace", takže
# zaškrtnutí "Ručně upravit pozice čepů" v režimu "Kontrola existujícího řešení"
# nemělo vůbec žádný efekt, přestože checkbox byl v obou režimech viditelný
# a aktivní. Ruční úprava čepů teď funguje v obou režimech.
if enable_manual_pin:
    st.title(f"🔧 {app_mode} (Ruční úprava čepů)")
    st.markdown("### 🎛️ Volitelné ladění pozic čepů a počáteční síly")
    
    min_x1 = max(0.0, Xb1 - L0_1)
    max_x1 = min(lid_length, Xb1 + L0_1)
    default_x1 = float(np.clip(default_lx1, min_x1, max_x1))
    
    lx1 = st.slider("Čep hlavní vzpěry – X na víku (mm)", min_value=float(min_x1), max_value=float(max_x1), value=default_x1, step=1.0)
    
    inner_val1 = L0_1**2 - (lx1 - Xb1)**2
    ly1 = Yb1 + np.sqrt(max(0.0, inner_val1))
    if ly1 > lid_height + 300 or ly1 < -300:
        ly1 = Yb1 - np.sqrt(max(0.0, inner_val1))

    if use_aux:
        st.markdown("---")
        st.markdown("#### Pomocná vzpěra – ruční nastavení X a Y")
        col_p1, col_p2 = st.columns(2)
        lx2 = col_p1.slider("Čep pomocné vzpěry – X (mm)", min_value=-400.0, max_value=float(lid_length + 200.0), value=float(default_lx2), step=1.0)
        ly2 = col_p2.slider("Čep pomocné vzpěry – Y (mm)", min_value=-300.0, max_value=float(lid_height + 300.0), value=float(default_ly2), step=1.0)
        
        geom_L2_0 = np.sqrt((lx2 - Xb2) ** 2 + (ly2 - Yb2) ** 2)
        st.info(
            f"💡 Mechanický rozsah vzpěry: **{L2_min_input:.0f}–{L2_min_input + S2:.0f} mm** "
            f"(zdvih {S2:.0f} mm) | aktuální geometrická vzdálenost čepů @0° = **{geom_L2_0:.1f} mm**"
        )
    else:
        lx2, ly2 = 0.0, 0.0

    st.markdown("---")
    force_offset_at_0 = st.slider("Cílená síla na madlu při otevření @0° (N) [kladná = tlačit, záporná = drží samo]", -200.0, 200.0, 0.0, 5.0)
else:
    st.title(f"🔧 {app_mode}")
    lx1, ly1 = default_lx1, default_ly1
    lx2, ly2 = default_lx2, default_ly2
    # OPRAVA: dřív tu bylo "L2_min_input = 478.0", což natvrdo přepsalo hodnotu
    # zadanou uživatelem v sidebaru. L2_min_input už je správně nastavené výše
    # (buď ze sidebar number_input, nebo na výchozí hodnotu, pokud use_aux == False),
    # takže se sem už nesahá.

F_main, F_aux = solve_forces_smooth(lx1, ly1, lx2, ly2, target_force_at_0=force_offset_at_0 if enable_manual_pin else None)

# Kontrola, jestli navržená/ručně zadaná geometrie nepřekračuje mechanický zdvih pomocné vzpěry
if use_aux:
    L2_travel_min, L2_travel_max, deg_at_min, deg_at_max = aux_strut_travel_mm(lx2, ly2, theta_max)
    L2_allowed_max = L2_min_input + S2
    tol = 0.5
    used_stroke = L2_travel_max - L2_travel_min

    # Jediná platná podmínka: geometrická vzdálenost čepů musí být KDEKOLI na dráze
    # (včetně theta=0°) uvnitř mechanického rozsahu vzpěry [L2_min_input, L2_min_input+S2].
    # Není potřeba, aby se rovnala L2_min_input přesně při theta=0 - to je jen dolní mez.
    if L2_travel_min < L2_min_input - tol or L2_travel_max > L2_allowed_max + tol:
        st.warning(
            f"⚠️ Pomocná vzpěra by během otevírání musela mít délku "
            f"{L2_travel_min:.0f}–{L2_travel_max:.0f} mm (min. při {deg_at_min:.0f}°, max. při {deg_at_max:.0f}°), "
            f"ale její mechanický rozsah je jen {L2_min_input:.0f}–{L2_allowed_max:.0f} mm (zdvih {S2:.0f} mm). "
            "V části rozsahu by narazila na doraz – upravte polohu čepů/vany nebo zdvih vzpěry."
        )
    elif used_stroke < 0.5 * S2:
        st.info(
            f"ℹ️ Pomocná vzpěra na této dráze reálně využije jen {used_stroke:.0f} mm ze svého "
            f"{S2:.0f} mm zdvihu ({L2_travel_min:.0f}–{L2_travel_max:.0f} mm). "
            "Zvažte jinou polohu vany/čepu pro lepší využití zdvihu."
        )

def calc_F_hand_internal(th, Fm_nom, Fa_nom):
    d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, th)
    d2 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, th) if use_aux else 0.0
    Fm_act, Fa_act = get_struts_forces_at_angle(th, Fm_nom, Fa_nom, lx1, ly1, lx2, ly2)
    Ts_main = n_main * Fm_act * d1
    Ts_aux = n_aux * Fa_act * d2 if use_aux else 0.0
    h_arm = handle_moment_arm_m(th)
    return -(Tg(th) + Ts_main + Ts_aux) / h_arm

def F_hand(theta):
    return calc_F_hand_internal(theta, F_main, F_aux)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Zadaná síla hlavní vzpěry - 1ks", f"{F_main:.0f} N")
c2.metric("Zadaná síla pomocné vzpěry - 1ks" if use_aux else "Pomocná vzpěra", f"{F_aux:.0f} N" if use_aux else "—")
c3.metric("Potřebná síla k otevření @0°", f"{F_hand(0.0):.1f} N")
f_max_val = F_hand(theta_max)
c4.metric("Potřebná síla k zavření", f"{f_max_val:.1f} N", "Drží víko" if f_max_val < 0 else "tlačí ven")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Čep na víku – hlavní X", f"{lx1:.1f} mm")
c6.metric("Čep na víku – hlavní Y", f"{ly1:.1f} mm")
c7.metric("Čep na víku – pomocná X", f"{lx2:.1f} mm" if use_aux else "—")
c8.metric("Čep na víku – pomocná Y", f"{ly2:.1f} mm" if use_aux else "—")

st.divider()

col_geo, col_force = st.columns(2)
fig1, ax1 = plt.subplots(figsize=(5, 3.1))
fig2, ax2 = plt.subplots(figsize=(5, 3.1))
common_adjust = {'left': 0.16, 'bottom': 0.18, 'right': 0.95, 'top': 0.9}
fig1.subplots_adjust(**common_adjust)
fig2.subplots_adjust(**common_adjust)

def draw_geometry_mm(ax, theta):
    ax.clear()
    corners_local = [(0, 0), (lid_length, 0), (lid_length, lid_height), (0, lid_height)]
    corners_global = [rotate_mm(lx, ly, theta) for lx, ly in corners_local]
    xs = [p[0] for p in corners_global] + [corners_global[0][0]]
    ys = [p[1] for p in corners_global] + [corners_global[0][1]]
    
    ax.fill(xs, ys, color="#c9a876", alpha=0.6, edgecolor="black", linewidth=0.75, zorder=3)
    ax.plot(0, 0, "ko", markersize=4, zorder=5)
    ax.annotate("Pant", (0, 0), textcoords="offset points", xytext=(-8, -12), fontsize=9, fontweight='bold')

    Xc, Yc = rotate_mm(cg_x_mm, cg_y_mm, theta)
    ax.plot(Xc, Yc, "o", color="red", markersize=4.5, zorder=6)
    ax.annotate("CG", (Xc, Yc), textcoords="offset points", xytext=(6, 6), color="red", fontsize=9, fontweight='bold')

    hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
    ax.plot(hx, hy, "go", markersize=4.5, zorder=7)
    ax.annotate("Madlo", (hx, hy), textcoords="offset points", xytext=(6, 6), color="green", fontsize=9, fontweight='bold')

    Xp1, Yp1 = rotate_mm(lx1, ly1, theta)
    ax.plot([Xb1, Xp1], [Yb1, Yp1], "-", color="#1f77b4", linewidth=1.5, zorder=4, label="Hlavní vzpěra")
    ax.plot(Xb1, Yb1, "s", color="#1f77b4", markersize=3.5, zorder=5)
    ax.plot(Xp1, Yp1, "^", color="#1f77b4", markersize=3.5, zorder=5)

    if use_aux:
        Xp2, Yp2 = rotate_mm(lx2, ly2, theta)
        ax.plot([Xb2, Xp2], [Yb2, Yp2], "-", color="#d62728", linewidth=1.5, zorder=4, label="Pomocná vzpěra")
        ax.plot(Xb2, Yb2, "s", color="#d62728", markersize=3.5, zorder=5)
        ax.plot(Xp2, Yp2, "^", color="#d62728", markersize=3.5, zorder=5)

    all_x = xs + [0, Xb1] + ([Xb2] if use_aux else []) + ([Xp1] if 'Xp1' in locals() else [])
    all_y = ys + [0, Yb1] + ([Yb2] if use_aux else []) + ([Yp1] if 'Yp1' in locals() else [])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y, 800.0)
    
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.6, alpha=0.5)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.4, alpha=0.5)
    
    ax.set_box_aspect(1)
    ax.invert_xaxis()
    ax.tick_params(axis='both', labelsize=8)
    
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment('right')

    ax.set_title(f"Geometrie @ {np.degrees(theta):.1f}°", fontsize=10, fontweight='bold')
    ax.legend(loc="upper left", fontsize=7)

def draw_force_profile(ax, theta_marker=None):
    ax.clear()
    ax.set_xlim(0, 110)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

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
        ax.plot(np.degrees(theta_marker), F_hand(theta_marker), "o", color="black", markersize=4, zorder=6)

    ax.set_box_aspect(1)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlabel("Úhel otevření (°)", fontsize=9)
    ax.set_ylabel("Síla na madlu (N)", fontsize=9)
    ax.set_title("Profil síly na madlu", fontsize=10, fontweight='bold')
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.3, which='both')

theta_disp = np.radians(st.session_state.anim_deg)
draw_geometry_mm(ax1, theta_disp)
draw_force_profile(ax2, theta_disp)
col_geo.pyplot(fig1)
col_force.pyplot(fig2)

if st.session_state.is_playing:
    if st.session_state.anim_deg >= theta_max_deg:
        st.session_state.anim_deg = 0.0
    else:
        st.session_state.anim_deg += 2.0
        if st.session_state.anim_deg > theta_max_deg:
            st.session_state.anim_deg = float(theta_max_deg)
    time.sleep(0.04)
    st.rerun()
