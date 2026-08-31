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
        @page { size: A4 landscape; margin: 10mm; }
        header, footer, [data-testid="stSidebar"], [data-testid="stHeader"], .stTabs, button { display: none !important; }
        body { background-color: white; color: black; }
    }
    .print-btn {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
        cursor: pointer;
        margin-bottom: 1rem;
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

def find_max_angle_single(Xb, Yb, lx, ly, L_ext):
    def obj(theta):
        Xp, Yp = rotate_mm(lx, ly, theta)
        return distance_mm(Xb, Yb, Xp, Yp) - L_ext
    try:
        max_rad = brentq(obj, 0.0, np.radians(180))
        return max_rad
    except ValueError:
        return None

def find_max_angle_dual(Xb1, Yb1, lx1, ly1, L_ext1, Xb2, Yb2, lx2, ly2, L_ext2):
    best_rad = np.radians(180)
    rad1 = find_max_angle_single(Xb1, Yb1, lx1, ly1, L_ext1)
    if rad1 is not None and rad1 > 0:
        best_rad = min(best_rad, rad1)
    rad2 = find_max_angle_single(Xb2, Yb2, lx2, ly2, L_ext2)
    if rad2 is not None and rad2 > 0:
        best_rad = min(best_rad, rad2)
    return best_rad if best_rad < np.radians(180) else None

# ----------------------------------------------------------------------
# UI - SIDEBAR (Pouze parametry víka, těžiště a madla)
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
    st.title("🧮 Vzpěrovač (2× Hlavní vzpěra)")
    
    if st.button("🖨️ Vytisknout / Uložit do PDF", key="print_t1"):
        st.markdown("<script>window.print();</script>", unsafe_allow_html=True)
    
    col_single = st.columns(1)[0]
    with col_single:
        st.markdown("### Hlavní vzpěra (2 ks)")
        strut_type = st.selectbox("Typ hlavní vzpěry", list(STRUT_DATA.keys()), index=4, key="t1_type")
        S1 = st.number_input("Zdvih vzpěry (mm)", 10.0, 1500.0, 300.0, 5.0, key="t1_s")
        F_nom = st.number_input("Jmenovitá síla vzpěry F1 (N / 1ks)", 10.0, 20000.0, 650.0, 10.0, key="t1_f")
        
        fitting_type = st.selectbox("Koncovky hlavní vzpěry", list(FITTING_DATA.keys()), index=2, key="t1_fit")
        if fitting_type == "Vlastní (ruční zadání)":
            k1 = st.number_input("Koncovka vana (mm)", 0.0, 200.0, 18.0, key="t1_k1")
            k2 = st.number_input("Koncovka víko (mm)", 0.0, 200.0, 18.0, key="t1_k2")
        else:
            k1 = k2 = FITTING_DATA[fitting_type]
            
        Xb1 = st.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0, key="t1_xb")
        Yb1 = st.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0, key="t1_yb")
        lx1 = st.number_input("Víko X hlavní (mm)", -1000.0, 3000.0, 369.0, 5.0, key="t1_lx")
        ly1 = st.number_input("Víko Y hlavní (mm)", -1000.0, 1000.0, 233.0, 5.0, key="t1_ly")

    # Výpočty pro záložku 1
    offset = STRUT_DATA[strut_type]["offset"]
    progression = STRUT_DATA[strut_type]["progression"]

    L_ext = (2.0 * S1) + offset + k1 + k2
    L_com = L_ext - S1

    L_geom_0 = distance_mm(Xb1, Yb1, lx1, ly1)
    valid_geometry = (L_com <= L_geom_0 <= L_ext)

    st.divider()

    if valid_geometry:
        st.success(f"✅ Geometrie souhlasí! Zadané čepy jsou od sebe {L_geom_0:.1f} mm v zavřeném stavu, což je bezpečně v rozsahu vzpěry ({L_com:.1f} až {L_ext:.1f} mm).")
    else:
        st.error(f"⚠️ POZOR KOLIZE! Zadané čepy jsou od sebe {L_geom_0:.1f} mm. Tento rozměr je mimo povolený rozsah vzpěry ({L_com:.1f} až {L_ext:.1f} mm).")

    theta_max_rad = find_max_angle_single(Xb1, Yb1, lx1, ly1, L_ext)

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

    def get_actual_strut_force(L_current, l_ext_val, s_val, f_nom_val, prog_val):
        compression_ratio = np.clip((l_ext_val - L_current) / s_val, 0.0, 1.0)
        return f_nom_val * (1.0 + prog_val * compression_ratio)

    def F_hand(theta):
        Xp1, Yp1 = rotate_mm(lx1, ly1, theta)
        L_current = distance_mm(Xb1, Yb1, Xp1, Yp1)
        
        Fm_act = get_actual_strut_force(L_current, L_ext, S1, F_nom, progression)
        d1 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta)
        
        Ts_main = 2 * Fm_act * d1
        h_arm = handle_moment_arm_m(theta)
        
        return -(Tg(theta) + Ts_main) / h_arm

    col_plots, col_table = st.columns([2, 1])

    with col_plots:
        theta_disp_deg = st.slider("Zobrazit geometrii pro úhel (°)", 0.0, float(theta_max_deg), 0.0, step=1.0, key="t1_slider")
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
        custom_ang = st.number_input("Zadej úhel (°)", 0.0, float(theta_max_deg), default_custom_ang, 1.0, key="t1_cust")
        custom_f = F_hand(np.radians(custom_ang))
        color = "green" if custom_f < 0 else "red"
        st.markdown(f"Síla při **{custom_ang}°**: <span style='color:{color}; font-size: 1.2em; font-weight:bold;'>{custom_f:.1f} N</span>", unsafe_allow_html=True)

with tab2:
    st.title("🧮 Vzpěrovač (Hlavní + Pomocná vzpěra)")
    
    if st.button("🖨️ Vytisknout / Uložit do PDF", key="print_t2"):
        st.markdown("<script>window.print();</script>", unsafe_allow_html=True)
        
    col_m, col_a = st.columns(2)
    with col_m:
        st.markdown("### Hlavní vzpěra (2 ks)")
        strut_type_m = st.selectbox("Typ hlavní vzpěry", list(STRUT_DATA.keys()), index=4, key="t2_m_type")
        S_m = st.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 300.0, 5.0, key="t2_m_s")
        F_nom_m = st.number_input("Jmenovitá síla hlavní vzpěry F1 (N / 1ks)", 10.0, 20000.0, 650.0, 10.0, key="t2_m_f")
        
        fit_m = st.selectbox("Koncovky hlavní vzpěry", list(FITTING_DATA.keys()), index=2, key="t2_m_fit")
        if fit_m == "Vlastní (ruční zadání)":
            k1_m = st.number_input("Koncovka vana (mm)", 0.0, 200.0, 18.0, key="t2_m_k1")
            k2_m = st.number_input("Koncovka víko (mm)", 0.0, 200.0, 18.0, key="t2_m_k2")
        else:
            k1_m = k2_m = FITTING_DATA[fit_m]
            
        Xb_m = st.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0, key="t2_m_xb")
        Yb_m = st.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0, key="t2_m_yb")
        lx_m = st.number_input("Víko X hlavní (mm)", -1000.0, 3000.0, 369.0, 5.0, key="t2_m_lx")
        ly_m = st.number_input("Víko Y hlavní (mm)", -1000.0, 1000.0, 233.0, 5.0, key="t2_m_ly")

    with col_a:
        st.markdown("### Pomocná vzpěra (2 ks)")
        strut_type_a = st.selectbox("Typ pomocné vzpěry", list(STRUT_DATA.keys()), index=2, key="t2_a_type")
        S_a = st.number_input("Zdvih pomocné vzpěry (mm)", 10.0, 1500.0, 150.0, 5.0, key="t2_a_s")
        F_nom_a = st.number_input("Jmenovitá síla pomocné vzpěry F1 (N / 1ks)", 10.0, 20000.0, 300.0, 10.0, key="t2_a_f")
        
        fit_a = st.selectbox("Koncovky pomocné vzpěry", list(FITTING_DATA.keys()), index=2, key="t2_a_fit")
        if fit_a == "Vlastní (ruční zadání)":
            k1_a = st.number_input("Koncovka vana pomocná (mm)", 0.0, 200.0, 18.0, key="t2_a_k1")
            k2_a = st.number_input("Koncovka víko pomocná (mm)", 0.0, 200.0, 18.0, key="t2_a_k2")
        else:
            k1_a = k2_a = FITTING_DATA[fit_a]
            
        Xb_a = st.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 200.0, 5.0, key="t2_a_xb")
        Yb_a = st.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -150.0, 5.0, key="t2_a_yb")
        lx_a = st.number_input("Víko X pomocná (mm)", -1000.0, 3000.0, 150.0, 5.0, key="t2_a_lx")
        ly_a = st.number_input("Víko Y pomocná (mm)", -1000.0, 1000.0, 100.0, 5.0, key="t2_a_ly")

    # Výpočty pro záložku 2
    off_m = STRUT_DATA[strut_type_m]["offset"]
    prog_m = STRUT_DATA[strut_type_m]["progression"]
    L_ext_m = (2.0 * S_m) + off_m + k1_m + k2_m
    L_com_m = L_ext_m - S_m

    off_a = STRUT_DATA[strut_type_a]["offset"]
    prog_a = STRUT_DATA[strut_type_a]["progression"]
    L_ext_a = (2.0 * S_a) + off_a + k1_a + k2_a
    L_com_a = L_ext_a - S_a

    dist_0_m = distance_mm(Xb_m, Yb_m, lx_m, ly_m)
    dist_0_a = distance_mm(Xb_a, Yb_a, lx_a, ly_a)

    valid_m = (L_com_m <= dist_0_m <= L_ext_m)
    valid_a = (L_com_a <= dist_0_a <= L_ext_a)

    st.divider()
    if valid_m and valid_a:
        st.success("✅ Geometrie obou vzpěr v zavřeném stavu je v platném rozsahu zdvihu.")
    else:
        if not valid_m:
            st.error(f"⚠️ Hlavní vzpěra je mimo rozsah! Aktuální délka při 0° je {dist_0_m:.1f} mm (povolené rozmezí: {L_com_m:.1f} až {L_ext_m:.1f} mm).")
        if not valid_a:
            st.error(f"⚠️ Pomocná vzpěra je mimo rozsah! Aktuální délka při 0° je {dist_0_a:.1f} mm (povolené rozmezí: {L_com_a:.1f} až {L_ext_a:.1f} mm).")

    theta_max_rad_2 = find_max_angle_dual(Xb_m, Yb_m, lx_m, ly_m, L_ext_m, Xb_a, Yb_a, lx_a, ly_a, L_ext_a)

    if theta_max_rad_2 is None or theta_max_rad_2 <= 0:
        st.error("❌ Geometrie kombinace vzpěr neumožňuje otevření.")
    else:
        theta_max_deg_2 = np.degrees(theta_max_rad_2)
        st.info(f"Maximální úhel otevření pro kombinaci vzpěr: **{theta_max_deg_2:.1f}°**")

        def Tg_dual(theta):
            return -lid_mass * G * (cg_xm * np.cos(theta) - cg_ym * np.sin(theta))

        def F_hand_dual(theta):
            Xp_m, Yp_m = rotate_mm(lx_m, ly_m, theta)
            L_cur_m = distance_mm(Xb_m, Yb_m, Xp_m, Yp_m)
            ratio_m = np.clip((L_ext_m - L_cur_m) / S_m, 0.0, 1.0)
            Fm_act = F_nom_m * (1.0 + prog_m * ratio_m)
            d_m = signed_moment_arm_mm(Xb_m, Yb_m, lx_m, ly_m, theta)
            
            Xp_a, Yp_a = rotate_mm(lx_a, ly_a, theta)
            L_cur_a = distance_mm(Xb_a, Yb_a, Xp_a, Yp_a)
            ratio_a = np.clip((L_ext_a - L_cur_a) / S_a, 0.0, 1.0)
            Fa_act = F_nom_a * (1.0 + prog_a * ratio_a)
            d_a = signed_moment_arm_mm(Xb_a, Yb_a, lx_a, ly_a, theta)
            
            Ts_total = (2 * Fm_act * d_m) + (2 * Fa_act * d_a)
            h_arm = handle_moment_arm_m(theta)
            
            return -(Tg_dual(theta) + Ts_total) / h_arm

        col_p2, col_t2 = st.columns([2, 1])
        with col_p2:
            th_disp_2_deg = st.slider("Zobrazit geometrii (Hlavní + Pomocná) pro úhel (°)", 0.0, float(theta_max_deg_2), 0.0, step=1.0, key="t2_slider")
            th_disp_2 = np.radians(th_disp_2_deg)
            
            fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(10, 4.5))
            
            corners_local = [(0, 0), (lid_length, 0), (lid_length, lid_height), (0, lid_height)]
            corners_global = [rotate_mm(px, py, th_disp_2) for px, py in corners_local]
            xs_2 = [p[0] for p in corners_global] + [corners_global[0][0]]
            ys_2 = [p[1] for p in corners_global] + [corners_global[0][1]]
            
            ax2_1.fill(xs_2, ys_2, color="#c9a876", alpha=0.6, edgecolor="black")
            ax2_1.plot(0, 0, "ko", markersize=5, label="Pant")
            
            Xc_2, Yc_2 = rotate_mm(cg_x_mm, cg_y_mm, th_disp_2)
            ax2_1.plot(Xc_2, Yc_2, "ro", markersize=5, label="Těžiště")
            
            hx_2, hy_2 = rotate_mm(handle_x_mm, handle_y_mm, th_disp_2)
            ax2_1.plot(hx_2, hy_2, "go", markersize=5, label="Madlo")
            
            Xpm_2, Ypm_2 = rotate_mm(lx_m, ly_m, th_disp_2)
            ax2_1.plot([Xb_m, Xpm_2], [Yb_m, Ypm_2], "-", color="#1f77b4", linewidth=2, label="Hlavní vzpěra")
            ax2_1.plot(Xb_m, Yb_m, "s", color="#1f77b4")
            ax2_1.plot(Xpm_2, Ypm_2, "^", color="#1f77b4")
            
            Xpa_2, Ypa_2 = rotate_mm(lx_a, ly_a, th_disp_2)
            ax2_1.plot([Xb_a, Xpa_2], [Yb_a, Ypa_2], "-", color="#d62728", linewidth=2, label="Pomocná vzpěra")
            ax2_1.plot(Xb_a, Yb_a, "s", color="#d62728")
            ax2_1.plot(Xpa_2, Ypa_2, "^", color="#d62728")
            
            all_x2 = xs_2 + [0, Xb_m, Xpm_2, Xb_a, Xpa_2]
            all_y2 = ys_2 + [0, Yb_m, Ypm_2, Yb_a, Ypa_2]
            mid_x2, mid_y2 = (min(all_x2) + max(all_x2))/2, (min(all_y2) + max(all_y2))/2
            max_range2 = max(max(all_x2)-min(all_x2), max(all_y2)-min(all_y2), 800)
            
            ax2_1.set_xlim(mid_x2 - max_range2*0.55, mid_x2 + max_range2*0.55)
            ax2_1.set_ylim(mid_y2 - max_range2*0.55, mid_y2 + max_range2*0.55)
            ax2_1.set_box_aspect(1)
            ax2_1.invert_xaxis()
            ax2_1.grid(alpha=0.4)
            ax2_1.set_title(f"Geometrie @ {th_disp_2_deg:.1f}°")
            ax2_1.legend(fontsize=7)
            
            thetas_2 = np.linspace(0, theta_max_rad_2, 100)
            forces_n_2 = np.array([F_hand_dual(t) for t in thetas_2])
            degs_2 = np.degrees(thetas_2)
            
            ax2_2.axhline(0, color="black", linewidth=1)
            ax2_2.fill_between(degs_2, forces_n_2, 0, where=(forces_n_2 >= 0), color="#ff7f0e", alpha=0.5, label="Nutno tlačit")
            ax2_2.fill_between(degs_2, forces_n_2, 0, where=(forces_n_2 < 0), color="#2ca02c", alpha=0.5, label="Drží samo / tlačí ven")
            ax2_2.plot(degs_2, forces_n_2, color="black", linewidth=1.5)
            ax2_2.plot(th_disp_2_deg, F_hand_dual(th_disp_2), "ko", markersize=6)
            
            ax2_2.set_box_aspect(1)
            ax2_2.set_xlabel("Úhel otevření (°)")
            ax2_2.set_ylabel("Síla na madlu (N)")
            ax2_2.set_title("Profil síly na madlu")
            ax2_2.grid(alpha=0.4)
            ax2_2.legend(fontsize=8)
            
            st.pyplot(fig2)

        with col_t2:
            st.markdown("### Tabulka sil (Hlavní + Pomocná)")
            angles_to_check_2 = list(np.arange(0, theta_max_deg_2, 10.0))
            if angles_to_check_2[-1] != theta_max_deg_2:
                angles_to_check_2.append(theta_max_deg_2)
                
            data_2 = []
            for a in angles_to_check_2:
                f_val = F_hand_dual(np.radians(a))
                status = "Tlačí ven" if f_val < 0 else "Padá (nutno zvedat)"
                data_2.append({"Úhel (°)": f"{a:.1f}°", "Síla na madlu (N)": f"{f_val:.1f}", "Stav": status})
                
            df_2 = pd.DataFrame(data_2)
            st.dataframe(df_2, use_container_width=True, hide_index=True)
            
            st.markdown("### Kontrola vlastního úhlu")
            default_custom_ang_2 = min(45.0, float(theta_max_deg_2))
            custom_ang_2 = st.number_input("Zadej úhel (°)", 0.0, float(theta_max_deg_2), default_custom_ang_2, 1.0, key="t2_cust")
            custom_f_2 = F_hand_dual(np.radians(custom_ang_2))
            color_2 = "green" if custom_f_2 < 0 else "red"
            st.markdown(f"Síla při **{custom_ang_2}°**: <span style='color:{color_2}; font-size: 1.2em; font-weight:bold;'>{custom_f_2:.1f} N</span>", unsafe_allow_html=True)
