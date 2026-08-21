import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač PRO")
st.title("Vzpěrovač - CAD Synchronizovaný výpočet (Pevné délky)")

# --- UI - BOČNÍ PANEL ---
st.sidebar.header("1. Parametry víka")
m = st.sidebar.number_input("Hmotnost víka (kg)", value=30.0, step=1.0)
L_lid = st.sidebar.number_input("Délka víka (mm)", value=1000.0, step=10.0)
H_lid = st.sidebar.number_input("Výška/Tloušťka víka (mm)", value=150.0, step=10.0)
C_x = st.sidebar.number_input("Těžiště X (mm od pantu)", value=500.0, step=10.0)
C_y = st.sidebar.number_input("Těžiště Y (mm od pantu)", value=75.0, step=10.0)
C_0 = np.array([C_x, C_y])
max_angle = st.slider("Max. úhel otevření (°)", 45, 110, value=80)

pocet_vzper_radio = st.sidebar.radio("Konfigurace vzpěr", ["Pouze 2 hlavní", "2 hlavní + 2 pomocné (zadní)"], index=1)
pocet_vzper = 4 if "pomocné" in pocet_vzper_radio else 2

st.sidebar.header("2. Hlavní vzpěry (Přední)")
B_x1 = st.sidebar.number_input("Hlavní - vana X", value=520.0, step=10.0)
B_y1 = st.sidebar.number_input("Hlavní - vana Y", value=-126.0, step=10.0)
B1 = np.array([B_x1, B_y1])
L_closed1 = st.sidebar.number_input("Hlavní - Zasunutá délka (mm)", value=618.0, step=10.0)
stroke1 = st.sidebar.number_input("Hlavní - Zdvih (mm)", value=500.0, step=10.0)

if pocet_vzper == 4:
    st.sidebar.header("3. Pomocné vzpěry (Zadní u pantu)")
    B_x2 = st.sidebar.number_input("Pomocná - vana X", value=175.0, step=10.0)
    B_y2 = st.sidebar.number_input("Pomocná - vana Y", value=-301.0, step=10.0)
    B2 = np.array([B_x2, B_y2])
    L_closed2 = st.sidebar.number_input("Pomocná - Zasunutá délka (mm)", value=560.0, step=10.0)
    stroke2 = st.sidebar.number_input("Pomocná - Zdvih (mm)", value=120.0, step=10.0)
else:
    B2 = np.array([0.0, 0.0])
    L_closed2, stroke2 = 0.0, 0.0

# --- MATEMATIKA A GEOMETRIE ---
def rotate(pt, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        pt[0] * np.cos(a) - pt[1] * np.sin(a),
        pt[0] * np.sin(a) + pt[1] * np.cos(a)
    ])

def get_lid_mount(B, L_closed, stroke, angle_open):
    L_open = L_closed + stroke
    B_rot = rotate(B, -angle_open)
    d = np.linalg.norm(B_rot - B)
    if d > (L_closed + L_open) or d < abs(L_closed - L_open) or d == 0:
        return np.array([300.0, 200.0])
    a_val = (L_closed**2 - L_open**2 + d**2) / (2 * d)
    val = L_closed**2 - a_val**2
    h = np.sqrt(max(0, val))
    P2 = B + a_val * (B_rot - B) / d
    candidates = [
        np.array([P2[0] + h * (B_rot[1] - B[1]) / d, P2[1] - h * (B_rot[0] - B[0]) / d]),
        np.array([P2[0] - h * (B_rot[1] - B[1]) / d, P2[1] + h * (B_rot[0] - B[0]) / d])
    ]
    valid = [p for p in candidates if -50 <= p[0] <= L_lid * 1.5 and p[1] >= -50]
    return max(valid, key=lambda p: p[0]) if valid else candidates[0]

# 1. Přesný výpočet úhlu, kdy těžiště přechází přes svislou osu pantu (X_CG_global == 0)
angles_fine = np.linspace(0, 90, 1000)
cg_x_coords = np.array([rotate(C_0, a)[0] for a in angles_fine])
cg_zero_idx = np.argmin(np.abs(cg_x_coords))
alpha_cg_over = angles_fine[cg_zero_idx]

# 2. Čepy na víku
P1 = get_lid_mount(B1, L_closed1, stroke1, max_angle)

if pocet_vzper == 4:
    # Využijeme zadanou L_closed2 jako poloměr pro průchod pantem v okamžiku přechodu těžiště
    v_dir = np.array([0.0, 0.0]) - B2
    v_len = np.linalg.norm(v_dir)
    if v_len > 0:
        u_dir = v_dir / v_len
        P_dead_global = np.array([0.0, 0.0]) + u_dir * L_closed2
        P2 = rotate(P_dead_global, -alpha_cg_over)
    else:
        P2 = get_lid_mount(B2, L_closed2, stroke2, max_angle)
else:
    P2 = np.array([0.0, 0.0])

angles = np.linspace(0, max_angle, 100)
M_grav = m * 9.81 * (np.array([rotate(C_0, a)[0] for a in angles])) / 1000.0

def get_kinematics(P, B):
    d_arms, L_act = [], []
    for a in angles:
        Pt = rotate(P, a)
        vec = Pt - B
        L = np.linalg.norm(vec)
        L_act.append(L)
        d_arms.append((Pt[0]*vec[1] - Pt[1]*vec[0]) / (L * 1000.0))
    return np.array(d_arms), np.array(L_act)

d1, L1 = get_kinematics(P1, B1)
d2, L2 = get_kinematics(P2, B2) if pocet_vzper == 4 else (np.zeros_like(angles), np.zeros_like(angles))

# Výpočet sil vzpěr
target_hand_force_0 = 5.0 # kg
target_moment_0 = target_hand_force_0 * 9.81 * (L_lid / 1000.0)
req_M_0 = M_grav[0] - target_moment_0

if pocet_vzper == 4:
    F2 = 300.0 
    if np.abs(d2[0]) > 0.01:
        rem_M = req_M_0 - (F2 * 2 * d2[0])
        F1 = max(50.0, rem_M / (2 * d1[0]))
    else:
        F1 = 300.0
else:
    F2 = 0.0
    F1 = max(50.0, req_M_0 / (2 * d1[0]))

F1_rounded = np.ceil(max(0, F1) / 50.0) * 50
F2_rounded = np.ceil(max(0, F2) / 50.0) * 50 if pocet_vzper == 4 else 0.0

M_rear = (F2_rounded * 2) * d2 if pocet_vzper == 4 else np.zeros_like(angles)
M_front = (F1_rounded * 2) * d1
F_user = (M_grav - (M_front + M_rear)) / (L_lid / 1000.0) / 9.81

alpha_dead = alpha_cg_over if pocet_vzper == 4 else 0.0

# --- VÝSTUPNÍ METRIKY S X a Y ---
st.success("✅ Geometrie synchronizována se zadanými délkami!")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Hlavní vzpěra (1ks)", f"{F1_rounded:.0f} N")
col2.metric("Přední čep víko", f"X: {P1[0]:.0f}, Y: {max(0, P1[1]):.0f} mm")

if pocet_vzper == 4:
    col3.metric("Pomocná vzpěra (1ks)", f"{F2_rounded:.0f} N")
    col4.metric("Zadní čep víko", f"X: {P2[0]:.0f}, Y: {max(0, P2[1]):.0f} mm")
else:
    col3.metric("Pomocná vzpěra", "Není")
    col4.metric("Mrtvý bod", "Není")

col5.metric("Síla do ruky (Zavřeno)", f"{F_user[0]:.1f} kg")

st.divider()

# --- VIZUALIZACE ---
st.subheader("Animace víka, těžiště (CG) a profil síly do ruky")
curr_angle = st.slider("🔍 Úhel otevření víka", 0.0, float(max_angle), 0.0, step=1.0)
idx = int((curr_angle / max_angle) * 99)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Nákres geometrie
ax1.plot(0, 0, 'ko', markersize=8, label='Pant [0,0]')
lid_poly = np.array([[0, 0], [L_lid, 0], [L_lid, H_lid], [0, H_lid]])
lid_rot = np.array([rotate(pt, curr_angle) for pt in lid_poly])
polygon = Polygon(lid_rot, closed=True, fill=True, facecolor='gray', alpha=0.3, edgecolor='black', lw=2)
ax1.add_patch(polygon)

# Červené těžiště CG
cg_curr = rotate(C_0, curr_angle)
ax1.plot(*cg_curr, 'ro', markersize=10, label=f'Těžiště CG [{cg_curr[0]:.0f}, {cg_curr[1]:.0f}]')

# Hlavní vzpěra
Pt1 = rotate(P1, curr_angle)
ax1.plot(*B1, 'bs', markersize=6)
ax1.plot([B1[0], Pt1[0]], [B1[1], Pt1[1]], 'b-', lw=3, label=f'Hlavní ({L1[idx]:.0f} mm)')
ax1.plot(*Pt1, 'bo', markersize=6)

# Pomocná vzpěra
if pocet_vzper == 4:
    Pt2 = rotate(P2, curr_angle)
    ax1.plot(*B2, 'rs', markersize=6)
    ax1.plot([B2[0], Pt2[0]], [B2[1], Pt2[1]], 'r-', lw=2, label=f'Pomocná ({L2[idx]:.0f} mm)')
    ax1.plot(*Pt2, 'ro', markersize=6)

ax1.set_aspect('equal')
ax1.set_title(f"Geometrie modelu ({curr_angle:.1f}°)")
ax1.legend(loc="lower right")
ax1.grid(True, linestyle=':')
max_r = max(L_lid, H_lid, abs(B_x1), abs(B_y1)) * 1.1
ax1.set_xlim(-max_r*0.2, max_r)
ax1.set_ylim(-max_r*0.2, max_r*1.1)
ax1.invert_xaxis()

# 2. Graf síly do ruky
ax2.plot(angles, F_user, 'b-', lw=2)
ax2.axhline(0, color='black', lw=1)
ax2.plot(curr_angle, F_user[idx], 'ro', markersize=10, label=f"Nyní: {F_user[idx]:.1f} kg")
ax2.fill_between(angles, 0, F_user, where=(F_user >= 0), facecolor='red', alpha=0.2, label="Nutno zvedat (kladná)")
ax2.fill_between(angles, 0, F_user, where=(F_user < 0), facecolor='green', alpha=0.2, label="Drží / brzdit (záporná)")

if pocet_vzper == 4 and alpha_dead > 0:
    ax2.axvline(alpha_dead, color='orange', linestyle='--', lw=2, label=f'Mrtvý bod (přechod CG): {alpha_dead:.1f}°')

ax2.set_xlabel("Úhel otevření (°)")
ax2.set_ylabel("Síla potřebná na víku (kg)")
ax2.set_title("Profil síly do ruky")
ax2.legend()
ax2.grid(True, linestyle=':')

st.pyplot(fig)
