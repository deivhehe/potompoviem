import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def solve_main_pin_mm(Xb_mm, Yb_mm, L0_mm, S_mm, theta_max, L_lid_mm, H_lid_mm):
    def objective(v):
        lx, ly = v
        l1 = (lx - Xb_mm) ** 2 + (ly - Yb_mm) ** 2 - L0_mm ** 2
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        l2 = (Xp2 - Xb_mm) ** 2 + (Yp2 - Yb_mm) ** 2 - (L0_mm + S_mm) ** 2
        return l1**2 + l2**2
    res = minimize(objective, [L_lid_mm * 0.5, H_lid_mm * 0.5], bounds=[(0.0, L_lid_mm), (0.0, H_lid_mm)], method='L-BFGS-B')
    if res.success and res.fun < 1.0:
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


# Vytvoření záložek v aplikaci
tab_design, tab_control = st.tabs(["🔧 Návrh a optimalizace", "📊 Kontrola existujícího řešení"])

# ======================================================================
# ZÁLOŽKA 1: NÁVRH A OPTIMALIZACE
# ======================================================================
with tab_design:
    st.sidebar.header("1) Geometrie a hmotnost víka")
    lid_length = st.sidebar.number_input("Délka víka (mm)", 50.0, 3000.0, 1109.0, 10.0, key="d_len")
    lid_height = st.sidebar.number_input("Výška / tloušťka víka (mm)", 10.0, 1000.0, 812.0, 5.0, key="d_hei")
    lid_mass = st.sidebar.number_input("Hmotnost víka (kg)", 0.1, 500.0, 180.0, 0.5, key="d_mas")

    st.sidebar.header("2) Pozice madla (v zavřeném stavu, od pantu)")
    handle_x_mm = st.sidebar.number_input("Madlo X (mm)", 0.0, 3000.0, lid_length, 5.0, key="d_hx")
    handle_y_mm = st.sidebar.number_input("Madlo Y (mm)", -500.0, 1000.0, float(lid_height * 0.5), 5.0, key="d_hy")

    st.sidebar.header("3) Těžiště víka (od pantu, v zavřeném stavu)")
    cg_x_mm = st.sidebar.number_input("Těžiště X (mm)", 0.0, 3000.0, float(np.clip(lid_length * 0.5, 0.0, 3000.0)), 5.0, key="d_cgx")
    cg_y_mm = st.sidebar.number_input("Těžiště Y (mm)", -500.0, 1000.0, float(np.clip(lid_height * 0.5, -500.0, 1000.0)), 5.0, key="d_cgy")

    st.sidebar.header("4) Rozsah otevření")
    theta_max_deg = st.sidebar.slider("Maximální úhel otevření (°)", 45, 130, 83, key="d_thmax")

    st.sidebar.header("5) Konfigurace vzpěr")
    config_type = st.sidebar.radio("Typ uspořádání", ["2× hlavní vzpěra", "2× hlavní + 2× pomocná vzpěra"], key="d_cfg")
    use_aux = (config_type == "2× hlavní + 2× pomocná vzpěra")

    if use_aux:
        st.sidebar.header("6) Poměr sil vzpěr")
        force_ratio = st.sidebar.slider("Podíl síly hlavní vzpěry (%)", 10, 90, 50, 5, key="d_frat")
    else:
        force_ratio = 50

    st.sidebar.header("7) Hlavní vzpěra")
    Xb1 = st.sidebar.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0, key="d_xb1")
    Yb1 = st.sidebar.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0, key="d_yb1")
    L0_1 = st.sidebar.number_input("Zasunutá délka hlavní @0° (mm)", 30.0, 2000.0, 618.0, 5.0, key="d_l01")
    S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 500.0, 5.0, key="d_s1")

    if use_aux:
        st.sidebar.header("8) Pomocná vzpěra (konzolka)")
        Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0, key="d_xb2")
        Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0, key="d_yb2")
        L_min_2 = st.sidebar.number_input("Min. zasunutá délka pomocné vzpěry (mm)", 30.0, 2000.0, 500.0, 5.0, key="d_l02")
        S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm)", 10.0, 1500.0, 100.0, 5.0, key="d_s2")

    theta_disp_deg = st.sidebar.slider("Úhel pro geometrický náhled (°)", 0, theta_max_deg, 0, key="d_thdisp")
    animate = st.sidebar.button("▶️ Animovat otevírání", key="d_anim")

    # Výpočetní jádro záložky 1
    theta_max = np.radians(theta_max_deg)
    n_main = 2
    n_aux = 2

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

    pin1, ok1 = solve_main_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height)
    if not ok1:
        st.error("⚠️ Pro zadané parametry hlavní vzpěry nelze geometricky vyřešit pozici čepu.")
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
                ratio_actual = Fm / (Fm + Fa + 1e-6)
                ratio_target = force_ratio / 100.0
                ratio_penalty = (ratio_actual - ratio_target)**2 * 100000.0
            else:
                Fm = forces[0]
                if Fm < 10:
                    return 1e9
                Fa = 0.0
                ratio_penalty = 0.0
            
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

    def handle_moment_arm_m(theta):
        hx, hy = rotate_mm(handle_x_mm, handle_y_mm, theta)
        r = np.hypot(hx, hy)
        return r * 0.001 if r > 1e-6 else 1e-6

    def F_hand(theta):
        Ts_main = n_main * F_main * signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta)
        Ts_aux = n_aux * F_aux * signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, theta) if use_aux else 0.0
        h_arm = handle_moment_arm_m(theta)
        return -(Tg(theta) + Ts_main + Ts_aux) / h_arm

    st.title("🔧 Návrh plynových vzpěr víka")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Síla hlavní vzpěry (1 ks)", f"{F_main:.0f} N")
    c2.metric("Síla pomocné vzpěry (1 ks)", f"{F_aux:.0f} N" if use_aux else "—")
    c3.metric("Potřebná síla k otevření", f"{F_hand(0.0):.1f} N")
    f_max_val = F_hand(theta_max)
    c4.metric("Potřebná síla k zavření", f"{f_max_val:.1f} N", "Drží víko v otevřené pozici" if f_max_val < 0 else "tlačí ven")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Čep na víku – hlavní X", f"{lx1:.1f} mm")
    c6.metric("Čep na víku – hlavní Y", f"{ly1:.1f} mm")
    c7.metric("Čep na víku – pomocná X", f"{lx2:.1f} mm" if use_aux else "—")
    c8.metric("Čep na víku – pomocná Y", f"{ly2:.1f} mm" if use_aux else "—")

    if theta_dead is not None:
        st.info(f"🔹 Mrtvý bod při úhlu **{np.degrees(theta_dead):.1f}°**")
    st.divider()

    col_geo, col_force = st.columns(2)
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    common_adj = {'left': 0.2, 'bottom': 0.2, 'right': 0.95, 'top': 0.9}
    fig1.subplots_adjust(**common_adj)
    fig2.subplots_adjust(**common_adj)

    theta_disp = np.radians(theta_disp_deg)

    def draw_geo(ax, th):
        ax.clear()
        corners_local = [(0, 0), (lid_length, 0), (lid_length, lid_height), (0, lid_height)]
        corners_global = [rotate_mm(lx, ly, th) for lx, ly in corners_local]
        xs = [p[0] for p in corners_global] + [corners_global[0][0]]
        ys = [p[1] for p in corners_global] + [corners_global[0][1]]
        ax.fill(xs, ys, color="#c9a876", alpha=0.6, edgecolor="black", linewidth=1.5, zorder=3)
        ax.plot(0, 0, "ko", markersize=8, zorder=5)
        Xc, Yc = rotate_mm(cg_x_mm, cg_y_mm, th)
        ax.plot(Xc, Yc, "o", color="red", markersize=9, zorder=6)
        hx, hy = rotate_mm(handle_x_mm, handle_y_mm, th)
        ax.plot(hx, hy, "go", markersize=9, zorder=7)
        Xp1, Yp1 = rotate_mm(lx1, ly1, th)
        ax.plot([Xb1, Xp1], [Yb1, Yp1], "-", color="#1f77b4", linewidth=3, zorder=4)
        ax.plot(Xb1, Yb1, "s", color="#1f77b4", markersize=7, zorder=5)
        ax.plot(Xp1, Yp1, "^", color="#1f77b4", markersize=7, zorder=5)
        if use_aux:
            Xp2, Yp2 = rotate_mm(lx2, ly2, th)
            ax.plot([Xb2, Xp2], [Yb2, Yp2], "-", color="#d62728", linewidth=3, zorder=4)
            ax.plot(Xb2, Yb2, "s", color="#d62728", markersize=7, zorder=5)
            ax.plot(Xp2, Yp2, "^", color="#d62728", markersize=7, zorder=5)
        max_dim = max(lid_length, lid_height)
        ax.set_xlim(-max_dim * 0.25, lid_length * 1.2)
        ax.set_ylim(-400, max(lid_height * 1.5, 300))
        ax.set_box_aspect(1)
        ax.invert_xaxis()
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title(f"Geometrie víka @ {np.degrees(th):.1f}°", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

    def draw_prof(ax, th_m):
        ax.clear()
        thetas = np.linspace(0, theta_max, 200)
        forces_n = np.array([F_hand(t) for t in thetas])
        degs = np.degrees(thetas)
        ax.axhline(0, color="black", linewidth=1)
        ax.fill_between(degs, forces_n, 0, where=(forces_n >= 0), color="#ff7f0e", alpha=0.5)
        ax.fill_between(degs, forces_n, 0, where=(forces_n < 0), color="#2ca02c", alpha=0.5)
        ax.plot(degs, forces_n, color="black", linewidth=1.5)
        if theta_dead is not None:
            ax.axvline(np.degrees(theta_dead), color="purple", linestyle="--", linewidth=1.5)
        if th_m is not None:
            ax.plot(np.degrees(th_m), F_hand(th_m), "o", color="black", markersize=8, zorder=6)
        ax.set_box_aspect(1)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlabel("Úhel otevření (°)", fontsize=9)
        ax.set_ylabel("Síla na madlu (N)", fontsize=9)
        ax.set_title("Profil síly na madlu", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

    if animate:
        ph1, ph2 = col_geo.empty(), col_force.empty()
        for deg in np.linspace(0, theta_max_deg, 40):
            th = np.radians(deg)
            draw_geo(ax1, th)
            draw_prof(ax2, th)
            ph1.pyplot(fig1)
            ph2.pyplot(fig2)
            time.sleep(0.04)
    else:
        draw_geo(ax1, theta_disp)
        draw_prof(ax2, theta_disp)
        col_geo.pyplot(fig1)
        col_force.pyplot(fig2)


# ======================================================================
# ZÁLOŽKA 2: KONTROLA EXISTUJÍCÍHO ŘEŠENÍ
# ======================================================================
with tab_control:
    st.title("📊 Kontrola síly pro zadanou geometrii vzpěr")
    
    c_in1, c_in2 = st.columns(2)
    with c_in1:
        st.subheader("Parametry víka")
        c_len = st.number_input("Délka víka (mm)", 50.0, 3000.0, 1109.0, 10.0, key="c_len")
        c_hei = st.number_input("Výška / tloušťka víka (mm)", 10.0, 1000.0, 812.0, 5.0, key="c_hei")
        c_mas = st.number_input("Hmotnost víka (kg)", 0.1, 500.0, 180.0, 0.5, key="c_mas")
        c_thmax_deg = st.slider("Maximální úhel otevření (°)", 45, 130, 83, key="c_thmax")
        c_hx = st.number_input("Madlo X (mm)", 0.0, 3000.0, c_len, 5.0, key="c_hx")
        c_hy = st.number_input("Madlo Y (mm)", -500.0, 1000.0, float(c_hei * 0.5), 5.0, key="c_hy")
        c_cgx = st.number_input("Těžiště X (mm)", 0.0, 3000.0, float(c_len * 0.5), 5.0, key="c_cgx")
        c_cgy = st.number_input("Těžiště Y (mm)", -500.0, 1000.0, float(c_hei * 0.5), 5.0, key="c_cgy")

    with c_in2:
        st.subheader("Uložení vzpěr (čepy na vaně i na víku)")
        c_use_aux = st.checkbox("Zahrnout i pomocné vzpěry", value=True, key="c_useaux")
        
        st.markdown("**Hlavní vzpěra**")
        c_xb1 = st.number_input("Vana X hlavní (mm)", -1000.0, 3000.0, 585.0, 5.0, key="c_xb1")
        c_yb1 = st.number_input("Vana Y hlavní (mm)", -1000.0, 1000.0, -111.0, 5.0, key="c_yb1")
        c_lx1 = st.number_input("Čep na víku X hlavní (mm)", -500.0, 3000.0, 618.0, 5.0, key="c_lx1")
        c_ly1 = st.number_input("Čep na víku Y hlavní (mm)", -500.0, 1000.0, 400.0, 5.0, key="c_ly1")
        
        if c_use_aux:
            st.markdown("**Pomocná vzpěra**")
            c_xb2 = st.number_input("Vana X pomocná (mm)", -1000.0, 3000.0, 145.0, 5.0, key="c_xb2")
            c_yb2 = st.number_input("Vana Y pomocná (mm)", -1000.0, 1000.0, -241.0, 5.0, key="c_yb2")
            c_lx2 = st.number_input("Čep na víku X pomocná (mm)", -500.0, 3000.0, 100.0, 5.0, key="c_lx2")
            c_ly2 = st.number_input("Čep na víku Y pomocná (mm)", -500.0, 1000.0, 200.0, 5.0, key="c_ly2")

    # Výpočet pro kontrolní záložku
    c_th_max = np.radians(c_thmax_deg)
    c_cg_xm, c_cg_ym = c_cgx * 0.001, c_cgy * 0.001

    def c_Tg(theta):
        return -c_mas * G * (c_cg_xm * np.cos(theta) - c_cg_ym * np.sin(theta))

    def c_solve_required_forces():
        def objective(forces):
            if c_use_aux:
                Fm, Fa = forces
                if Fm < 0 or Fa < 0:
                    return 1e9
            else:
                Fm = forces[0]
                if Fm < 0:
                    return 1e9
                Fa = 0.0
            
            err = 0.0
            for th in np.linspace(0, c_th_max, 5):
                d1 = signed_moment_arm_mm(c_xb1, c_yb1, c_lx1, c_ly1, th)
                if c_use_aux:
                    d2 = signed_moment_arm_mm(c_xb2, c_yb2, c_lx2, c_ly2, th)
                    moment_vzpěr = 2 * Fm * d1 + 2 * Fa * d2
                else:
                    moment_vzpěr = 2 * Fm * d1
                moment_tíže = -c_Tg(th)
                err += (moment_vzpěr - moment_tíže)**2
            return err

        if c_use_aux:
            res = minimize(objective, [500.0, 500.0], bounds=[(0, 10000), (0, 10000)], method='L-BFGS-B')
            if res.success:
                return res.x[0], res.x[1]
            return 0.0, 0.0
        else:
            res = minimize(objective, [500.0], bounds=[(0, 10000)], method='L-BFGS-B')
            if res.success:
                return res.x[0], 0.0
            return 0.0, 0.0

    c_F_main, c_F_aux = c_solve_required_forces()

    def c_handle_arm(theta):
        hx, hy = rotate_mm(c_hx, c_hy, theta)
        r = np.hypot(hx, hy)
        return r * 0.001 if r > 1e-6 else 1e-6

    def c_F_hand(theta):
        Ts_main = 2 * c_F_main * signed_moment_arm_mm(c_xb1, c_yb1, c_lx1, c_ly1, theta)
        Ts_aux = 2 * c_F_aux * signed_moment_arm_mm(c_xb2, c_yb2, c_lx2, c_ly2, theta) if c_use_aux else 0.0
        h_arm = c_handle_arm(theta)
        return -(c_Tg(theta) + Ts_main + Ts_aux) / h_arm

    st.divider()
    st.subheader("Výsledky kontroly")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Potřebná síla hlavní vzpěry (1 ks)", f"{c_F_main:.0f} N")
    if c_use_aux:
        cc2.metric("Potřebná síla pomocné vzpěry (1 ks)", f"{c_F_aux:.0f} N")
    else:
        cc2.metric("Pomocná vzpěra", "—")
    cc3.metric("Potřebná síla k otevření @0°", f"{c_F_hand(0.0):.1f} N")
    c_fmax = c_F_hand(c_th_max)
    cc4.metric("Potřebná síla k zavření @max", f"{c_fmax:.1f} N", "Drží víko v otevřené pozici" if c_fmax < 0 else "tlačí ven")
