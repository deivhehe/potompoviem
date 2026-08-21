import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač")
st.title("Vzpěrovač - Kompletní CAD kontrolní panel")

# --- UI - NASTAVENÍ ---
st.sidebar.header("1. Parametry víka")
m = st.sidebar.number_input("Hmotnost víka (kg)", value=30.0, step=1.0)
L_lid = st.sidebar.number_input("Délka víka (mm)", value=1000.0, step=10.0)
H_lid = st.sidebar.number_input("Výška víka (mm)", value=800.0, step=10.0)
C_x = st.sidebar.number_input("Těžiště X (mm od pantu)", value=500.0)
C_y = st.sidebar.number_input("Těžiště Y (mm od pantu)", value=400.0)
max_angle = st.sidebar.slider("Max. úhel otevření (°)", 45, 110, value=80)

st.sidebar.header("2. Hlavní vzpěry")
B1 = np.array([st.sidebar.number_input("Vana X1", value=520.0), st.sidebar.number_input("Vana Y1", value=-126.0)])
P1 = np.array([st.sidebar.number_input("Víko X1", value=415.0), st.sidebar.number_input("Víko Y1", value=490.0)])
L_closed1 = st.sidebar.number_input("Zasunutá délka 1 (mm)", value=618.0)
stroke1 = st.sidebar.number_input("Zdvih 1 (mm)", value=500.0)

st.sidebar.header("3. Zadní vzpěry")
B2 = np.array([st.sidebar.number_input("Vana X2", value=175.0), st.sidebar.number_input("Vana Y2", value=-301.0)])
P2 = np.array([st.sidebar.number_input("Víko X2", value=150.0), st.sidebar.number_input("Víko Y2", value=580.0)])
L_closed2 = st.sidebar.number_input("Zasunutá délka 2 (mm)", value=560.0)
F2_user = st.sidebar.number_input("Síla zadní (N, 1ks)", value=150.0)

# --- VÝPOČTY ---
angles = np.linspace(0, max_angle, 100)
H = np.array([0.0, 0.0])

def rotate(pt, angle):
    a = np.radians(angle)
    return np.array([pt[0]*np.cos(a) - pt[1]*np.sin(a), pt[0]*np.sin(a) + pt[1]*np.cos(a)])

def get_kinematics(P, B):
    d_arms, L_act = [], []
    for a in angles:
        Pt = rotate(P, a)
        vec = Pt - B
        L = np.linalg.norm(vec)
        L_act.append(L)
        # Rameno síly = kolmá vzdálenost vektoru vzpěry od pantu
        d_arms.append((Pt[0]*vec[1] - Pt[1]*vec[0]) / (L * 1000.0))
    return np.array(d_arms), np.array(L_act)

d1, L1 = get_kinematics(P1, B1)
d2, L2 = get_kinematics(P2, B2)
M_grav = m * 9.81 * (np.array([rotate([C_x, C_y], a)[0] for a in angles])) / 1000.0

# Síla přední (1ks) vypočtená tak, aby víko v zavřeném stavu potřebovalo zvednout cca 5kg
target_hand_force = 5.0 
req_M_0 = M_grav[0] - (target_hand_force * 9.81 * L_lid/1000)
F1 = max(50.0, np.ceil(max(0, (req_M_0 - F2_user * 2 * d2[0]) / (2 * d1[0])) / 50) * 50)
F_user = (M_grav - (F1 * 2 * d1 + F2_user * 2 * d2)) / (L_lid / 1000.0) / 9.81
alpha_dead = angles[np.argmin(np.abs(d2))]

# --- METRIKY ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Hlavní vzpěra (1ks)", f"{F1:.0f} N")
col2.metric("Přední čep víko (X,Y)", f"[{P1[0]:.0f}, {P1[1]:.0f}]")
col3.metric("Zadní čep víko (X,Y)", f"[{P2[0]:.0f}, {P2[1]:.0f}]")
col4.metric("Síla (Zavřeno)", f"{F_user[0]:.1f} kg")
col5.metric("Mrtvý bod zadní", f"{alpha_dead:.1f}°")

# --- VIZUALIZACE ---
curr = st.slider("Animace úhlu", 0.0, float(max_angle), 0.0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Nákres
lid = Polygon(np.array([rotate(pt, curr) for pt in [[0,0], [L_lid,0], [L_lid,H_lid], [0,H_lid]]]), facecolor='gray', alpha=0.3)
ax1.add_patch(lid)
cg = rotate([C_x, C_y], curr)
ax1.plot(*cg, 'ro', markersize=12, label="Těžiště CG")
for P, B, col, lbl in [(P1, B1, 'b', 'Hlavní'), (P2, B2, 'r', 'Zadní')]:
    Pt = rotate(P, curr)
    ax1.plot([B[0], Pt[0]], [B[1], Pt[1]], col+'-', lw=3, label=lbl)
    ax1.plot(*B, col+'s'); ax1.plot(*Pt, col+'o')

ax1.set_aspect('equal'); ax1.set_xlim(-200, 1200); ax1.set_ylim(-400, 900); ax1.grid(True); ax1.legend()
ax1.invert_xaxis()

# Graf síly
ax2.plot(angles, F_user, 'g-', lw=2)
ax2.axhline(0, color='k'); ax2.axvline(alpha_dead, color='orange', linestyle='--')
ax2.plot(curr, F_user[int((curr/max_angle)*99)], 'ro', markersize=10)
ax2.set_title("Profil síly do ruky (kg)"); ax2.grid(True)
st.pyplot(fig)
