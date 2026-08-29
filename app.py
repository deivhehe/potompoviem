import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import brentq
import pandas as pd

st.set_page_config(page_title="Vzpěrovač", layout="wide")

st.markdown("""
    <style>
    @media print {
        @page { size: A4 landscape; margin: 0mm; }
        header, footer, [data-testid="stSidebar"], [data-testid="stHeader"] { display: none !important; }
    }
    </style>
""", unsafe_allow_html=True)

G = 9.81  # m/s^2

# ----------------------------------------------------------------------
# DATA Z TABULKY
# ----------------------------------------------------------------------
STRUT_DATA = {
    "Typ G3/8": {"offset": 32, "progression": 0.30},
    "Typ G4/12": {"offset": 32, "progression": 0.25},
    "Typ G6/15": {"offset": 34, "progression": 0.35},
    "Typ G8/19": {"offset": 53, "progression": 0.35},
    "Typ G10/23": {"offset": 60, "progression": 0.35},
    "Typ G14/28": {"offset": 68, "progression": 0.50},
    "Typ G20/40": {"offset": 116, "progression": 0.35},
    "Typ G22/40": {"offset": 32, "progression": 0.45},
    "Typ G25/55": {"offset": 140, "progression": 0.35},
    "Typ G30/65": {"offset": 160, "progression": 0.35}
}

FITTING_DATA = {
    "WX18 (18 mm)": 18.0,
    "WG22 / WX22 (22 mm)": 22.0,
    "WG18 (18 mm)": 18.0,
    "WG30 (30 mm)": 30.0,
    "WG35 (35 mm)": 35.0,
    "WG45 (45 mm)": 45.0,
    "Vlastní (ruční zadání)": None
}

# ----------------------------------------------------------------------
# MATEMATIKA A GEOMETRIE
# ----------------------------------------------------------------------
def rotate_mm(lx, ly, theta):
    c, s = np.cos(theta), np.sin(theta)
    return lx * c - ly * s, lx * s + ly * c

def distance_mm(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def signed_moment_arm_mm(Xb_mm, Yb_mm, lx_mm, ly_mm, theta):
    Xp_mm, Yp_mm = rotate_mm(lx_mm, ly_mm, theta)
    L_mm = distance_mm(Xb_mm, Yb_mm, Xp_mm, Yp_mm)
    if L_mm < 1e-6: return 0.0
    return (Xb_mm * Yp_mm - Yb_mm * Xp_mm) / (L_mm * 1000.0)

def find_max_angle(Xb, Yb, lx, ly, L_ext):
    def obj(theta):
        Xp, Yp = rotate_mm(lx, ly, theta)
        return distance_mm(Xb, Yb, Xp, Yp) - L_ext
    try:
        max_rad = brentq(obj, 0.0, np.radians(180))
        return max_rad
    except ValueError:
        return None

# ----------------------------------------------------------------------
# UI - SIDEBAR (Společné parametry víka, těžiště a madla)
# ----------------------------------------------------------------------
st.sidebar.header("1) Parametry víka")
lid_length = st.sidebar.number_input("Délka víka (mm)", 50.0, 3000.0, 1109.0, 10.0)
lid_height = st.sidebar.number_input("Výška / tloušťka víka (mm)", 10.0, 1000.0, 812.0, 5.0)
lid_mass = st.sidebar.number_input("Hmotnost víka (kg)", 0.1, 500.0, 180.0, 0.5)

st.sidebar.header("2) Těžiště a Madlo")
cg_x_mm = st.sidebar.number_input("Těžiště X (mm od pantu)", 0.0, 3000.0, float(np.clip(lid_length * 0.5, 0.0, 3000.0)), 5.0)
cg_y_mm = st.sidebar.number_input("Těžiště Y (mm od pantu)", -500.0, 1000.0, float(np.clip(lid_height * 0.5, -500.0, 1000.0)), 5.0)
handle_x_mm = st.sidebar.number_input("Madlo X (mm od pantu)", 0.0, 3000.0, lid_length, 5.0)
handle_y_mm = st.sidebar.number_input("Madlo Y (mm od pantu)", -500.0, 1000.0, float(lid_height * 0.5), 5.0)

# Hlavní přepínač záložek
tab1, tab2 = st.tabs(["2× Hlavní vzpěra", "Hlavní + Pomocná vzpěra"])

with tab1:
    st.sidebar.header("3) Výběr hlavní vzpěry")
    strut_type = st.sidebar.selectbox("Typ vzpěry", list(STRUT_DATA.keys()), index=4)
    S1 = st.sidebar.number_input("Zdvih vzpěry (mm)", 10.0, 1500.0, 300.0, 5.0)
    F_nom = st.sidebar.number_input("Jmenovitá síla vzpěry F1 (N / 1ks)", 10.0, 20000.0, 650.0, 10.0)

    st.sidebar.header("4) Koncovky")
    fitting_type = st.sidebar.selectbox("Typ koncovek", list(FITTING_DATA.keys()), index=2)
    if fitting_type == "Vlastní (ruční zadání)":
        k1 = st.sidebar.number_input("Délka koncovky na vaně (mm)", 0.0, 200.0, 18.0)
        k2 = st.sidebar.number_input("Délka koncovky na víku (mm)", 0.0, 200.0, 18.0)
    else:
        k1 = k2 = FITTING_DATA[fitting_type]

    st.sidebar.header("5) Pozice čepů (natvrdo)")
    Xb1 = st.sidebar.number_input("Vana - X (mm)", -1000.0, 3000.0, 585.0, 5.0)
    Yb1 = st.sidebar.number_input("Vana - Y (mm)", -1000.0, 1000.0, -111.0, 5.0)
    lx1 = st.sidebar.number_input("Víko - X (mm)", -1000.0, 3000.0, 369.0, 5.0)
    ly1 = st.sidebar.number_input("Víko - Y (mm)", -1000.0, 1000.0, 233.0, 5.0)

    # Výpočty pro záložku 1
    offset = STRUT_DATA[strut_type]["offset"]
    progression = STRUT_DATA[strut_type]["progression"]

    L_ext = (2.0 * S1) + offset + k1 + k2
    L_com = L_ext - S1

    L_geom_0 = distance_mm(Xb1, Yb1, lx1, ly1)
    diff = L_geom_0 - L_com

    st.title("🧮 Vzpěrovač")

    if abs(diff) <= 2.0:
        st.success(f"✅ Geometrie souhlasí! Stlačená vzpěra má {L_com:.1f} mm a zadané čepy jsou od sebe {L_geom_0:.1f} mm.")
    else:
        st.error(f"⚠️ POZOR KOLIZE! Stlačená vzpěra má délku {L_com:.1f} mm, ale zadané čepy jsou od sebe {L_geom_0:.1f} mm. Rozdíl je {diff:.1f} mm.")

    theta_max_rad = find_max_angle(Xb1, Yb1, lx1, ly1, L_ext)

    if theta_max_rad is None or theta_max_rad <= 0:
        st.error("❌ Geometrie neumožňuje otevření. Čepy se od sebe nikdy nevzdálí na roztaženou délku vzpěry, nebo jsou špatně zadané.")
        st.stop()

    theta_max_deg = np.degrees(theta_max_rad)

    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Maximální úhel otevření (doraz)", f"{theta_max_deg:.1f}°")
    col_i2.metric("Roztažená délka vzpěry", f"{L_ext:.1f} mm")
    col_i3.metric("Stlačená délka vzpěry", f"{L_com:.1f} mm")
    col_i4.metric("Síla ve stlačeném stavu (1ks)", f"{F_nom * (1 + progression):.0f} N")

    st.divider()

    cg_xm, cg_ym = cg_x_mm * 0.001, cg_y_mm * 0.001

    def Tg(theta):
        return -lid_mass * G * (cg_xm * np.cos(theta) - cg_ym * np.sin(theta))

    def handle_moment_arm_m(theta):
        hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
        r = np.hypot(hx, hy)
        return r * 0.001 if r > 1e-6 else 1e-6

    def get_actual_strut_force(L_current):
        compression_ratio = np.clip((L_ext - L_current) / S1, 0.0, 1.0)
        return F_nom * (1.0 + progression * compression_ratio)

    def F_hand(theta):
        Xp1, Yp1 = rotate_mm(lx1, ly1, theta)
        L_current = distance_mm(Xb1, Yb1, Xp1, Yp1)
        
        Fm_act = get_actual_strut_force(L_current)
        d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta)
        
        Ts_main = 2 * Fm_act * d1
        h_arm = handle_moment_arm_m(theta)
        
        return -(Tg(theta) + Ts_main) / h_arm

    col_plots, col_table = st.columns([2, 1])

    with col_plots:
        theta_disp_deg = st.slider("Zobrazit geometrii pro úhel (°)", 0.0, float(theta_max_deg), 0.0, step=1.0)
        theta_disp = np.radians(theta_disp_deg)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        
        corners_local = [(0, 0), (lid_length, 0), (lid_length, lid_height), (0, lid_height)]
        corners_global = [rotate_mm(px, py, theta_disp) for px, py in corners_local]
        xs = [p[0] for p in corners_global] + [corners_global[0][0]]
        ys = [p[1] for p in corners_global] + [corners_global[0][1]]
        
        ax1.fill(xs, ys, color="#c9a876", alpha=0.6, edgecolor="black")
        ax1.plot(0, 0, "ko", markersize=5, label="Pant")
        
        Xc, Yc = rotate_mm(cg_x_mm, cg_y_mm, theta_disp)
        ax1.plot(Xc, Yc, "ro", markersize=5, label="Těžiště")
        
        hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta_disp)
        ax1.plot(hx, hy, "go", markersize=5, label="Madlo")
        
        Xp1, Yp1 = rotate_mm(lx1, ly1, theta_disp)
        ax1.plot([Xb1, Xp1], [Yb1, Yp1], "-", color="#1f77b4", linewidth=2.5, label="Vzpěra")
        ax1.plot(Xb1, Yb1, "s", color="#1f77b4")
        ax1.plot(Xp1, Yp1, "^", color="#1f77b4")
        
        all_x = xs + [0, Xb1, Xp1]
        all_y = ys + [0, Yb1, Yp1]
        mid_x, mid_y = (min(all_x) + max(all_x))/2, (min(all_y) + max(all_y))/2
        max_range = max(max(all_x)-min(all_x), max(all_y)-min(all_y), 800)
        
        ax1.set_xlim(mid_x - max_range*0.55, mid_x + max_range*0.55)
        ax1.set_ylim(mid_y - max_range*0.55, mid_y + max_range*0.55)
        ax1.set_box_aspect(1)
        ax1.invert_xaxis()
        ax1.grid(alpha=0.4)
        ax1.set_title(f"Geometrie @ {theta_disp_deg:.1f}°")
        ax1.legend(fontsize=8)
        
        thetas = np.linspace(0, theta_max_rad, 100)
        forces_n = np.array([F_hand(t) for t in thetas])
        degs = np.degrees(thetas)
        
        ax2.axhline(0, color="black", linewidth=1)
        ax2.fill_between(degs, forces_n, 0, where=(forces_n >= 0), color="#ff7f0e", alpha=0.5, label="Nutno tlačit")
        ax2.fill_between(degs, forces_n, 0, where=(forces_n < 0), color="#2ca02c", alpha=0.5, label="Drží samo / tlačí ven")
        ax2.plot(degs, forces_n, color="black", linewidth=1.5)
        ax2.plot(theta_disp_deg, F_hand(theta_disp), "ko", markersize=6)
        
        ax2.set_box_aspect(1)
        ax2.set_xlabel("Úhel otevření (°)")
        ax2.set_ylabel("Síla na madlu (N)")
        ax2.set_title("Profil síly na madlu")
        ax2.grid(alpha=0.4)
        ax2.legend(fontsize=8)
        
        st.pyplot(fig)

    with col_table:
        st.markdown("### Tabulka sil")
        
        angles_to_check = list(np.arange(0, theta_max_deg, 10.0))
        if angles_to_check[-1] != theta_max_deg:
            angles_to_check.append(theta_max_deg)
            
        data = []
        for a in angles_to_check:
            f_val = F_hand(np.radians(a))
            status = "Tlačí ven" if f_val < 0 else "Padá (nutno zvedat)"
            data.append({"Úhel (°)": f"{a:.1f}°", "Síla na madlu (N)": f"{f_val:.1f}", "Stav": status})
            
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### Kontrola vlastního úhlu")
        default_custom_ang = min(45.0, float(theta_max_deg))
        custom_ang = st.number_input("Zadej úhel (°)", 0.0, float(theta_max_deg), default_custom_ang, 1.0)
        custom_f = F_hand(np.radians(custom_ang))
        color = "green" if custom_f < 0 else "red"
        st.markdown(f"Síla při **{custom_ang}°**: <span style='color:{color}; font-size: 1.2em; font-weight:bold;'>{custom_f:.1f} N</span>", unsafe_allow_html=True)

with tab2:
    st.title("🧮 Vzpěrovač (Hlavní + Pomocná vzpěra)")
    st.info("Tato záložka se právě připravuje pro výpočet kombinace hlavních a pomocných vzpěr s průběžnou kontrolou délky a polohy těžiště.")
