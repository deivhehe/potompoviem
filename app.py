import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač")
st.title("Vzpěrovač")

# --- VÝCHOZÍ HODNOTY ---
DEFAULT_VALUES = {
    "m": 30.0,
    "L_lid": 1000.0,
    "H_lid": 150.0,
    "C_x": 500.0,
    "C_y": 75.0,
    "max_angle": 80.0,
    "pocet_vzper": 4,
    "B_x1": 520.0,
    "B_y1": -126.0,
    "L_closed1": 618.0,
    "stroke1": 500.0,
    "B_x2": 175.0,
    "B_y2": -301.0,
    "L_closed2": 560.0,
    "stroke2": 120.0,
    "F2_user": 200.0
}

for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.sidebar.button("🔄 Resetovat do výchozího stavu"):
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    st.rerun()

# --- UŽIVATELSKÉ ROZHRANÍ ---
st.sidebar.header("1. Parametry víka")
m = st.sidebar.number_input("Hmotnost víka (kg)", step=1.0, key="m")
L_lid = st.sidebar.number_input("Délka víka (mm)", step=10.0, key="L_lid")
H_lid = st.sidebar.number_input("Výška/Tloušťka víka (mm)", step=10.0, key="H_lid")

C_x = st.sidebar.number_input("Těžiště osa X (mm od pantu)", step=10.0, key="C_x")
C_y = st.sidebar.number_input("Těžiště osa Y (mm od pantu)", step=10.0, key="C_y")
C_0 = np.array([C_x, C_y]) 
max_angle = st.sidebar.slider("Max. úhel otevření (°)", 45, 110, key="max_angle")

pocet_vzper_radio = st.sidebar.radio("Počet vzpěr celkem", [2, 4], index=0 if st.session_state["pocet_vzper"]==2 else 1, key="pocet_vzper_radio")
st.session_state["pocet_vzper"] = pocet_vzper_radio
pocet_vzper = st.session_state["pocet_vzper"]

# HLAVNÍ PÁR
st.sidebar.header("2. Hlavní vzpěry (Přední)")
st.sidebar.info("Referenční bod [0,0] je pant (vpravo). Kladné X znamená vzdálenost doleva do vany.")
B_x1 = st.sidebar.number_input("Hlavní - čep vana X", step=10.0, key="B_x1")
B_y1 = st.sidebar.number_input("Hlavní - čep vana Y", step=10.0, key="B_y1")
B1 = np.array([B_x1, B_y1])
L_closed1 = st.sidebar.number_input("Hlavní - Zasunutá délka (mm)", step=10.0, key="L_closed1")
stroke1 = st.sidebar.number_input("Hlavní - Zdvih (mm)", step=10.0, key="stroke1")
L_ext1 = L_closed1 + stroke1

# ASISTENČNÍ PÁR
if pocet_vzper == 4:
    st.sidebar.header("3. Asistenční vzpěry (Zadní u pantu)")
    B_x2 = st.sidebar.number_input("Zadní - čep vana X", step=10.0, key="B_x2")
    B_y2 = st.sidebar.number_input("Zadní - čep vana Y", step=10.0, key="B_y2")
    B2 = np.array([B_x2, B_y2])
    L_closed2 = st.sidebar.number_input("Zadní - Zasunutá délka (mm)", step=10.0, key="L_closed2")
    stroke2 = st.sidebar.number_input("Zadní - Zdvih (mm)", step=10.0, key="stroke2")
    L_ext2 = L_closed2 + stroke2
    F2_user = st.sidebar.number_input("Síla zadní vzpěry (N)", step=50.0, key="F2_user")
else:
    F2_user = 0
    B2 = np.array([0,0])
    L_ext2, L_closed2, stroke2 = 0, 0, 0

g = 9.81
H = np.array([0.0, 0.0]) # Pant

# --- MATEMATIKA ---
def rotate(pt, origin, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        origin[0] + (pt[0] - origin[0]) * np.cos(a) - (pt[1] - origin[1]) * np.sin(a),
        origin[1] + (pt[0] - origin[0]) * np.sin(a) + (pt[1] - origin[1]) * np.cos(a)
    ])

def find_lid_mount_flexible(B, L_closed_base, stroke, max_angle, H, is_rear=False):
    if not is_rear:
        L_closed = max(L_closed_base + 5.0, 20.0)
        L_open = L_closed_base + stroke
    else:
        # Zadní: v zavřeném vysunutá, v otevřeném zasunutá
        L_closed = L_closed_base + stroke
        L_open = max(L_closed_base + 5.0, 20.0)
    
    B_rot = rotate(B, H, -max_angle)
    d = np.linalg.norm(B_rot - B)
    
    # Velmi tolerantní rozsah pro průsečík kružnic, aby to nevyhazovalo chybu při ladění
    if d > (L_closed + L_open) + 30.0 or d < abs(L_closed - L_open) - 30.0 or d == 0:
        return None
    
    sum_r = L_closed + L_open
    if d > sum_r: d = sum_r
    diff_r = abs(L_closed - L_open)
    if d < diff_r: d = diff_r + 0.1

    a = (L_closed**2 - L_open**2 + d**2) / (2 * d)
    val = L_closed**2 - a**2
    h = np.sqrt(max(0, val))
    
    P2 = B + a * (B_rot - B) / d
    x3 = P2[0] + h * (B_rot[1] - B[1]) / d
    y3 = P2[1] - h * (B_rot[0] - B[0]) / d
    x4 = P2[0] - h * (B_rot[1] - B[1]) / d
    y4 = P2[1] + h * (B_rot[0] - B[0]) / d
    p3, p4 = np.array([x3, y3]), np.array([x4, y4])
    
    valid_points = [p for p in (p3, p4) if p[0] > -50 and p[1] >= -50.0]
    if not len(valid_points): return None
    elif len(valid_points) == 1: return valid_points[0]
    else:
        if is_rear:
            return valid_points[0] if valid_points[0][1] < valid_points[1][1] else valid_points[1]
        else:
            return valid_points[0] if valid_points[0][1] > valid_points[1][1] else valid_points[1]

# Výpočet čepů
P0_1 = find_lid_mount_flexible(B1, L_closed1, stroke1, max_angle, H, is_rear=False)
P0_2 = None
if pocet_vzper == 4:
    P0_2 = find_lid_mount_flexible(B2, L_closed2, stroke2, max_angle, H, is_rear=True)

if P0_1 is None:
    st.error("❌ Geometrické řešení pro HLAVNÍ vzpěru neexistuje. Upravte pozici čepu na vaně.")
elif pocet_vzper == 4 and P0_2 is None:
    st.error("❌ Geometrické řešení pro ZADNÍ vzpěru neexistuje. S touto zasunutou délkou (560 mm) a zdvihem (120 mm) posuňte zadní čep na vaně blíž k pantu nebo níže.")
else:
    angles = np.linspace(0, max_angle, 100)
    M_grav = m * g * (np.array([rotate(C_0, H, a)[0] for a in angles]) - H[0]) / 1000.0
    
    def get_kinematics(P0, B):
        d_arms, L_act = [], []
        for alpha in angles:
            P_t = rotate(P0, H, alpha)
            vec = P_t - B
            L = np.linalg.norm(vec)
            L_act.append(L)
            u = vec / L
            r = P_t - H
            d_arms.append((r[0]*u[1] - r[1]*u[0]) / 1000.0)
        return np.array(d_arms), np.array(L_act)

    d_arms1, L_act1 = get_kinematics(P0_1, B1)
    
    M_rear = np.zeros_like(angles)
    if pocet_vzper == 4:
        d_arms2, L_act2 = get_kinematics(P0_2, B2)
        M_rear = (F2_user * 2) * d_arms2

    valid_idx = np.abs(d_arms1) > 0.015 
    if np.any(valid_idx):
        req_M_front = (M_grav * 1.1) - M_rear
        F_1_total = np.max(req_M_front[valid_idx] / np.abs(d_arms1[valid_idx]))
    else:
        F_1_total = 0

    F_1_strut = max(0, F_1_total / 2)
    F_1_rounded = np.ceil(F_1_strut / 50.0) * 50

    M_front_act = (F_1_rounded * 2) * d_arms1
    M_net = M_front_act + M_rear - M_grav
    F_user_kg = (M_net / (L_lid / 1000.0)) / g

    st.success("✅ Geometrie kompletní!")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Přední vzpěra čep (X, Y)", f"[{P0_1[0]:.0f}, {max(0, P0_1[1]):.0f}]")
    col2.metric("Potřebná síla PŘEDNÍ", f"{F_1_rounded:.0f} N", "(1ks)")
    if pocet_vzper == 4:
        col3.metric("Zadní vzpěra čep (X, Y)", f"[{P0_2[0]:.0f}, {max(0, P0_2[1]):.0f}]")
    else:
        col3.metric("Zadní vzpěra", "Není")
    col4.metric("Síla do ruky - otevřeno", f"{-F_user_kg[-1]:.1f} kg")

    st.divider()

    # VIZUALIZACE
    current_angle = st.slider("🔍 Animace víka", 0.0, float(max_angle), 0.0, step=1.0)
    idx = int((current_angle / max_angle) * 99)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(*H, 'ko', markersize=8, label='Pant [0,0]')
    lid_poly = np.array([[0, 0], [L_lid, 0], [L_lid, H_lid], [0, H_lid]])
    lid_poly_rot = np.array([rotate(pt, H, current_angle) for pt in lid_poly])
    polygon = Polygon(lid_poly_rot, closed=True, fill=True, facecolor='gray', alpha=0.3, edgecolor='black', lw=2)
    ax1.add_patch(polygon)

    P_cur1 = rotate(P0_1, H, current_angle)
    ax1.plot(*B1, 'bs')
    ax1.plot([B1[0], P_cur1[0]], [B1[1], P_cur1[1]], 'b-', lw=4, label=f'Hlavní ({L_act1[idx]:.0f} mm)')
    ax1.plot(*P_cur1, 'bo')

    if pocet_vzper == 4:
        P_cur2 = rotate(P0_2, H, current_angle)
        ax1.plot(*B2, 'rs')
        ax1.plot([B2[0], P_cur2[0]], [B2[1], P_cur2[1]], 'r-', lw=2, label=f'Asistenční ({L_act2[idx]:.0f} mm)')
        ax1.plot(*P_cur2, 'ro')

    ax1.set_aspect('equal')
    ax1.set_title(f"Model ({current_angle:.1f}°) - Pant vpravo")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=':')
    max_r = max(L_lid, abs(B_x1), abs(B_y1)) * 1.1
    ax1.set_xlim(-max_r*0.2, max_r)
    ax1.set_ylim(-max_r*0.5, max_r)
    
    ax1.invert_xaxis() 

    ax2.plot(angles, -F_user_kg, 'b-', lw=2)
    ax2.axhline(0, color='black', lw=1)
    ax2.plot(current_angle, -F_user_kg[idx], 'ro', markersize=10)
    ax2.fill_between(angles, 0, -F_user_kg, where=(-F_user_kg >= 0), facecolor='red', alpha=0.2)
    ax2.fill_between(angles, 0, -F_user_kg, where=(-F_user_kg < 0), facecolor='green', alpha=0.2)
    
    if pocet_vzper == 4:
        cross_idx = np.where(np.diff(np.sign(d_arms2)))[0]
        if len(cross_idx) > 0:
            dead_ang = angles[cross_idx[0]]
            ax2.axvline(dead_ang, color='r', linestyle='--', label=f'Mrtvý úhel ({dead_ang:.0f}°)')

    ax2.set_xlabel("Úhel (°)")
    ax2.set_ylabel("Síla potřebná na víku (kg)")
    ax2.set_title("Síla do ruky")
    ax2.legend()
    ax2.grid(True, linestyle=':')

    st.pyplot(fig)
