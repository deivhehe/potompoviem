import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač")
st.title("Vzpěrovač - Výpočet plynových vzpěr")

# --- VÝCHOZÍ HODNOTY ---
DEFAULT_VALUES = {
    "m": 30.0,
    "L_lid": 1000.0,
    "H_lid": 150.0,
    "C_x": 500.0,
    "C_y": 75.0,
    "max_angle": 80.0,
    "pocet_vzper": 4,
    # Hlavní pár
    "B_x1": 400.0,
    "B_y1": -250.0,
    "P_x1": 415.0,
    "P_y1": 490.0,
    "L_closed1": 618.0,
    "stroke1": 500.0,
    # Asistenční pár (zadní)
    "B_x2": 175.0,
    "B_y2": -301.0,
    "P_x2": 150.0,
    "P_y2": 750.0,
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

# --- UŽIVATELSKé ROZHRANÍ ---
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
B_x1 = st.sidebar.number_input("Hlavní - čep vana X", step=10.0, key="B_x1")
B_y1 = st.sidebar.number_input("Hlavní - čep vana Y", step=10.0, key="B_y1")
P_x1 = st.sidebar.number_input("Hlavní - čep víko X", step=10.0, key="P_x1")
P_y1 = st.sidebar.number_input("Hlavní - čep víko Y", step=10.0, key="P_y1")
B1 = np.array([B_x1, B_y1])
P0_1 = np.array([P_x1, P_y1])
L_closed1 = st.sidebar.number_input("Hlavní - Zasunutá délka (mm)", step=10.0, key="L_closed1")
stroke1 = st.sidebar.number_input("Hlavní - Zdvih (mm)", step=10.0, key="stroke1")

# ASISTENČNÍ PÁR
if pocet_vzper == 4:
    st.sidebar.header("3. Asistenční vzpěry (Zadní)")
    B_x2 = st.sidebar.number_input("Zadní - čep vana X", step=10.0, key="B_x2")
    B_y2 = st.sidebar.number_input("Zadní - čep vana Y", step=10.0, key="B_y2")
    P_x2 = st.sidebar.number_input("Zadní - čep víko X", step=10.0, key="P_x2")
    P_y2 = st.sidebar.number_input("Zadní - čep víko Y", step=10.0, key="P_y2")
    B2 = np.array([B_x2, B_y2])
    P0_2 = np.array([P_x2, P_y2])
    L_closed2 = st.sidebar.number_input("Zadní - Zasunutá délka (mm)", step=10.0, key="L_closed2")
    stroke2 = st.sidebar.number_input("Zadní - Zdvih (mm)", step=10.0, key="stroke2")
    F2_user = st.sidebar.number_input("Síla zadní vzpěry (N, 1ks)", step=50.0, key="F2_user")
else:
    F2_user = 0
    B2, P0_2 = np.array([0,0]), np.array([0,0])
    L_closed2, stroke2 = 0, 0

g = 9.81
H = np.array([0.0, 0.0]) # Pant

# --- MATEMATIKA ---
angles = np.linspace(0, max_angle, 100)
M_grav = m * g * (np.array([np.cos(np.radians(a))*C_0[0] - np.sin(np.radians(a))*C_0[1] for a in angles]) - H[0]) / 1000.0

def rotate(pt, origin, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        origin[0] + (pt[0] - origin[0]) * np.cos(a) - (pt[1] - origin[1]) * np.sin(a),
        origin[1] + (pt[0] - origin[0]) * np.sin(a) + (pt[1] - origin[1]) * np.cos(a)
    ])

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

# Výpočet mrtvého úhlu zadní vzpěry (kde d_arms2 přechází přes nulu)
alpha_dead = 0.0
if pocet_vzper == 4:
    cross_idx = np.where(np.diff(np.sign(d_arms2)))[0]
    if len(cross_idx) > 0:
        alpha_dead = angles[cross_idx[0]]

valid_idx = np.abs(d_arms1) > 0.05 
if np.any(valid_idx):
    req_M_front = M_grav - M_rear
    F_1_strut = np.max(req_M_front[valid_idx] / np.abs(d_arms1[valid_idx])) / 2.0
else:
    F_1_strut = 0

F_1_rounded = max(50.0, np.ceil(max(0, F_1_strut) / 50.0) * 50)

M_front_act = (F_1_rounded * 2) * d_arms1
M_net = M_front_act + M_rear - M_grav
F_user_kg = (M_net / (L_lid / 1000.0)) / g

st.success("✅ Model načten podle zadaných čepů!")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hlavní vzpěra (1ks)", f"{F_1_rounded:.0f} N")
if pocet_vzper == 4:
    col2.metric("Mrtvý bod zadní vzpěry", f"{alpha_dead:.1f}°")
else:
    col2.metric("Zadní vzpěra", "Není osazena")
col3.metric("Síla do ruky (Zavřeno)", f"{-F_user_kg[0]:.1f} kg")
col4.metric("Síla do ruky (Otevřeno)", f"{-F_user_kg[-1]:.1f} kg")

st.divider()

# --- VIZUALIZACE A POSUVNÍK ---
st.subheader("Vizualizace a dráha víka")
current_angle = st.slider(
    f"🔍 Animace víka (Mrtvý bod zadní vzpěry: {alpha_dead:.1f}°)", 
    0.0, float(max_angle), 0.0, step=1.0
)

if pocet_vzper == 4:
    if current_angle < alpha_dead - 1.0:
        st.info(f"ℹ️ Úhel ({current_angle:.1f}°) je **před mrtvým bodem** – zadní vzpěra pomáhá zvedat.")
    elif current_angle > alpha_dead + 1.0:
        st.warning(f"⚠️ Úhel ({current_angle:.1f}°) je **za mrtvým bodem** – zadní vzpěra táhne dolů.")
    else:
        st.success(f"🎯 **Právě v mrtvém bodě ({alpha_dead:.1f}°)** – osa zadní vzpěry prochází osou pantu!")

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

ax2.plot(angles, -F_user_kg, 'b-', lw=2)
ax2.axhline(0, color='black', lw=1)
ax2.plot(current_angle, -F_user_kg[idx], 'ro', markersize=10, label=f"Nyní: {-F_user_kg[idx]:.1f} kg")
ax2.fill_between(angles, 0, -F_user_kg, where=(-F_user_kg >= 0), facecolor='red', alpha=0.2, label="Víko padá")
ax2.fill_between(angles, 0, -F_user_kg, where=(-F_user_kg < 0), facecolor='green', alpha=0.2, label="Víko drží samo")

if pocet_vzper == 4 and alpha_dead > 0:
    ax2.axvline(alpha_dead, color='orange', linestyle='--', lw=2, label=f'Mrtvý bod ({alpha_dead:.1f}°)')

ax2.set_xlabel("Úhel otevření (°)")
ax2.set_ylabel("Síla potřebná na víku (kg)")
ax2.set_title("Profil síly do ruky")
ax2.legend()
ax2.grid(True, linestyle=':')

st.pyplot(fig)
