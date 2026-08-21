import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač")
st.title("Vzpěrovač - Opravený profil síly")

# --- VÝCHOZÍ HODNOTY ---
DEFAULT_VALUES = {
    "m": 30.0,
    "L_lid": 1000.0,
    "H_lid": 150.0,
    "C_x": 500.0,
    "C_y": 75.0,
    "max_angle": 80.0,
    "pocet_vzper": 4,
    "B_x1": 650.0,
    "B_y1": -120.0,
    "L_closed1": 618.0,
    "stroke1": 500.0,
    "B_x2": 175.0,
    "B_y2": -301.0,
    "L_closed2": 560.0,
    "stroke2": 120.0,
    "F2_user": 200.0,
    "target_dead_angle": 45.0
}

for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.sidebar.button("🔄 Resetovat do výchozího stavu"):
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    st.rerun()

# --- UŽIVATELSKé ROZHRANÍ ---
st.sidebar.header("1. Parametry víka")
m = st.sidebar.number_input("Hmotnost víka (kg)", step=1.0, key="m")
L_lid = st.sidebar.number_input("Délka víka (mm)", step=10.0, key="L_lid")
H_lid = st.sidebar.number_input("Výška/Tloušťka víka (mm)", step=10.0, key="H_lid")

C_x = st.sidebar.number_input("Těžiště osa X (mm od pantu)", step=10.0, key="C_x")
C_y = st.sidebar.number_input("Těžiště osa Y (mm od pantu)", step=10.0, key="C_y")
C_0 = np.array([C_x, C_y]) 
max_angle = st.sidebar.slider("Max. úhel otevření (°)", 45, 110, key="max_angle")

target_dead_angle = st.sidebar.slider("Cílený mrtvý bod zadní vzpěry (°)", 20.0, 70.0, key="target_dead_angle")

pocet_vzper_radio = st.sidebar.radio("Počet vzpěr celkem", [2, 4], index=0 if st.session_state["pocet_vzper"]==2 else 1, key="pocet_vzper_radio")
st.session_state["pocet_vzper"] = pocet_vzper_radio
pocet_vzper = st.session_state["pocet_vzper"]

# HLAVNÍ PÁR
st.sidebar.header("2. Hlavní vzpěry (Přední)")
B_x1 = st.sidebar.number_input("Hlavní - čep vana X", step=10.0, key="B_x1")
B_y1 = st.sidebar.number_input("Hlavní - čep vana Y", step=10.0, key="B_y1")
B1 = np.array([B_x1, B_y1])
L_closed1 = st.sidebar.number_input("Hlavní - Zasunutá délka (mm)", step=10.0, key="L_closed1")
stroke1 = st.sidebar.number_input("Hlavní - Zdvih (mm)", step=10.0, key="stroke1")

# ASISTENČNÍ PÁR
if pocet_vzper == 4:
    st.sidebar.header("3. Asistenční vzpěry (Zadní)")
    B_x2 = st.sidebar.number_input("Zadní - čep vana X", step=10.0, key="B_x2")
    B_y2 = st.sidebar.number_input("Zadní - čep vana Y", step=10.0, key="B_y2")
    B2 = np.array([B_x2, B_y2])
    L_closed2 = st.sidebar.number_input("Zadní - Zasunutá délka (mm)", step=10.0, key="L_closed2")
    stroke2 = st.sidebar.number_input("Zadní - Zdvih (mm)", step=10.0, key="stroke2")
    F2_user = st.sidebar.number_input("Síla zadní vzpěry (N, 1ks)", step=50.0, key="F2_user")
else:
    F2_user = 0
    B2 = np.array([0,0])
    L_closed2, stroke2 = 0, 0

g = 9.81
H = np.array([0.0, 0.0]) # Pant

def rotate(pt, origin, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        origin[0] + (pt[0] - origin[0]) * np.cos(a) - (pt[1] - origin[1]) * np.sin(a),
        origin[1] + (pt[0] - origin[0]) * np.sin(a) + (pt[1] - origin[1]) * np.cos(a)
    ])

def get_lid_mount(B, L_closed, stroke, max_angle, H, is_rear=False, target_ang=45.0):
    if not is_rear:
        L_open = L_closed + stroke
        B_rot = rotate(B, H, -max_angle)
        d = np.linalg.norm(B_rot - B)
        if d > (L_closed + L_open) or d < abs(L_closed - L_open) or d == 0:
            return np.array([300.0, H_lid])
        a = (L_closed**2 - L_open**2 + d**2) / (2 * d)
        val = L_closed**2 - a**2
        h = np.sqrt(max(0, val))
        P2 = B + a * (B_rot - B) / d
        x3 = P2[0] + h * (B_rot[1] - B[1]) / d
        y3 = P2[1] - h * (B_rot[0] - B[0]) / d
        x4 = P2[0] - h * (B_rot[1] - B[1]) / d
        y4 = P2[1] + h * (B_rot[0] - B[0]) / d
        candidates = [np.array([x3, y3]), np.array([x4, y4])]
        valid = [p for p in candidates if -50 <= p[0] <= L_lid * 1.5 and p[1] >= -20]
        return max(valid, key=lambda p: p[0]) if valid else candidates[0]
    else:
        v_dir = H - B
        v_len = np.linalg.norm(v_dir)
        if v_len > 0:
            u_dir = v_dir / v_len
            P_dead_global = H + u_dir * L_closed
            return rotate(P_dead_global, H, -target_ang)
        return np.array([100.0, 50.0])

P0_1 = get_lid_mount(B1, L_closed1, stroke1, max_angle, H, is_rear=False)
P0_2 = np.array([0.0, 0.0])
if pocet_vzper == 4:
    P0_2 = get_lid_mount(B2, L_closed2, stroke2, max_angle, H, is_rear=True, target_ang=target_dead_angle)

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

d_arms2 = np.zeros_like(angles)
L_act2 = np.zeros_like(angles)
if pocet_vzper == 4:
    d_arms2, L_act2 = get_kinematics(P0_2, B2)

alpha_dead = target_dead_angle
if pocet_vzper == 4:
    cross_idx = np.where(np.diff(np.sign(d_arms2)))[0]
    if len(cross_idx) > 0:
        alpha_dead = angles[cross_idx[0]]

# Výpočet potřebné síly přední vzpěry tak, aby síla do ruky byla reálná a kladná v 0°
# Moment od vzpěr musí být menší než gravitační moment (aby víko samo nepadalo, ale museli jsme ho zvedat)
# M_net = M_grav - (M_front + M_rear) -> Síla do ruky > 0 v zavřeném stavu
target_hand_force_closed = 7.0 # kg
target_moment_closed = target_hand_force_closed * g * (L_lid / 1000.0)
req_M_front_0 = M_grav[0] - target_moment_closed
if pocet_vzper == 4:
    req_M_front_0 -= (F2_user * 2) * d_arms2[0]

F_1_strut = 300.0
if np.abs(d_arms1[0]) > 0.01:
    F_1_strut = max(50.0, req_M_front_0 / (2.0 * d_arms1[0]))

F_1_rounded = np.ceil(F_1_strut / 50.0) * 50

M_front_act = (F_1_rounded * 2) * d_arms1
M_rear = (F2_user * 2) * d_arms2 if pocet_vzper == 4 else np.zeros_like(angles)

# Celkový moment: Gravitace táhne dolů (+), vzpěry tlačí nahoru (-)
M_net = M_grav - (M_front_act + M_rear)
F_user_kg = (M_net / (L_lid / 1000.0)) / g

st.success("✅ Model přepočítán se správným znaménkem síly!")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hlavní vzpěra (1ks)", f"{F_1_rounded:.0f} N")
if pocet_vzper == 4:
    col2.metric("Zadní vzpěra (1ks)", f"{F2_user:.0f} N")
else:
    col2.metric("Zadní vzpěra", "Není")
col3.metric("Síla do ruky (Zavřeno)", f"{F_user_kg[0]:.1f} kg")
col4.metric("Síla do ruky (Otevřeno)", f"{F_user_kg[-1]:.1f} kg")

st.divider()

# --- VIZUALIZACE ---
st.subheader("Vizualizace a dráha víka")
current_angle = st.slider("🔍 Animace víka", 0.0, float(max_angle), 0.0, step=1.0)
idx = int((current_angle / max_angle) * 99)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(*H, 'ko', markersize=8, label='Pant [0,0]')
lid_poly = np.array([[0, 0], [L_lid, 0], [L_lid, H_lid], [0, H_lid]])
lid_poly_rot = np.array([rotate(pt, H, current_angle) for pt in lid_poly])
polygon = Polygon(lid_poly_rot, closed=True, fill=True, facecolor='gray', alpha=0.3, edgecolor='black', lw=2)
ax1.add_patch(polygon)

P_cur1 = rotate(P0_1, H, current_angle)
ax1.plot(*B1, 'bs', markersize=6)
ax1.plot([B1[0], P_cur1[0]], [B1[1], P_cur1[1]], 'b-', lw=4, label=f'Hlavní ({L_act1[idx]:.0f} mm)')
ax1.plot(*P_cur1, 'bo', markersize=6)

if pocet_vzper == 4:
    P_cur2 = rotate(P0_2, H, current_angle)
    ax1.plot(*B2, 'rs', markersize=6)
    ax1.plot([B2[0], P_cur2[0]], [B2[1], P_cur2[1]], 'r-', lw=2, label=f'Asistenční ({L_act2[idx]:.0f} mm)')
    ax1.plot(*P_cur2, 'ro', markersize=6)

ax1.set_aspect('equal')
ax1.set_title(f"Model ({current_angle:.1f}°) - Pant vpravo")
ax1.legend(loc="lower right")
ax1.grid(True, linestyle=':')
max_r = max(L_lid, H_lid, abs(B_x1), abs(B_y1)) * 1.1
ax1.set_xlim(-max_r*0.2, max_r)
ax1.set_ylim(-max_r*0.2, max_r*1.1)

ax1.invert_xaxis() 

ax2.plot(angles, F_user_kg, 'b-', lw=2)
ax2.axhline(0, color='black', lw=1)
ax2.plot(current_angle, F_user_kg[idx], 'ro', markersize=10, label=f"Nyní: {F_user_kg[idx]:.1f} kg")
ax2.fill_between(angles, 0, F_user_kg, where=(F_user_kg >= 0), facecolor='red', alpha=0.2, label="Nutno zvedat (kladná síla)")
ax2.fill_between(angles, 0, F_user_kg, where=(F_user_kg < 0), facecolor='green', alpha=0.2, label="Drží samo / brzdit (záporná)")

if pocet_vzper == 4 and alpha_dead > 0:
    ax2.axvline(alpha_dead, color='orange', linestyle='--', lw=2, label=f'Mrtvý bod ({alpha_dead:.1f}°)')

ax2.set_xlabel("Úhel otevření (°)")
ax2.set_ylabel("Síla potřebná na víku (kg)")
ax2.set_title("Profil síly do ruky")
ax2.legend()
ax2.grid(True, linestyle=':')

st.pyplot(fig)
