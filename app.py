import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač")
st.title("Vzpěrovač - Manuální nastavení čepů")

# --- PARAMETRY ---
if 'init' not in st.session_state:
    st.session_state.update({
        "m": 30.0, "L_lid": 1000.0, "H_lid": 800.0, "C_x": 500.0, "C_y": 400.0,
        "max_angle": 80.0, "B_x1": 520.0, "B_y1": -126.0, "P_x1": 415.0, "P_y1": 490.0,
        "L_closed1": 618.0, "stroke1": 500.0, "B_x2": 175.0, "B_y2": -301.0,
        "P_x2": 150.0, "P_y2": 580.0, "L_closed2": 560.0, "stroke2": 120.0, "F2_user": 150.0
    })
    st.session_state.init = True

# --- UI ---
st.sidebar.header("1. Parametry víka")
m = st.sidebar.number_input("Hmotnost (kg)", value=st.session_state.m)
C_x = st.sidebar.number_input("Těžiště X (mm)", value=st.session_state.C_x)
C_y = st.sidebar.number_input("Těžiště Y (mm)", value=st.session_state.C_y)
max_angle = st.sidebar.slider("Max. úhel (°)", 45, 110, value=80)

st.sidebar.header("2. Hlavní vzpěra")
B1 = np.array([st.sidebar.number_input("Vana X1", value=st.session_state.B_x1), st.sidebar.number_input("Vana Y1", value=st.session_state.B_y1)])
P1 = np.array([st.sidebar.number_input("Víko X1", value=st.session_state.P_x1), st.sidebar.number_input("Víko Y1", value=st.session_state.P_y1)])
L_closed1 = st.sidebar.number_input("Délka zavřené (mm)", value=st.session_state.L_closed1)
stroke1 = st.sidebar.number_input("Zdvih (mm)", value=st.session_state.stroke1)

st.sidebar.header("3. Zadní vzpěra")
B2 = np.array([st.sidebar.number_input("Vana X2", value=st.session_state.B_x2), st.sidebar.number_input("Vana Y2", value=st.session_state.B_y2)])
P2 = np.array([st.sidebar.number_input("Víko X2", value=st.session_state.P_x2), st.sidebar.number_input("Víko Y2", value=st.session_state.P_y2)])
L_closed2 = st.sidebar.number_input("Délka zavřené (mm) ", value=st.session_state.L_closed2)
F2_user = st.sidebar.number_input("Síla zadní (N)", value=st.session_state.F2_user)

# --- VÝPOČET ---
angles = np.linspace(0, max_angle, 100)
H = np.array([0.0, 0.0])
def rotate(pt, angle):
    a = np.radians(angle)
    return np.array([pt[0]*np.cos(a) - pt[1]*np.sin(a), pt[0]*np.sin(a) + pt[1]*np.cos(a)])

# Kinematika
def get_forces(P, B, F_strut):
    d_arms, L_act = [], []
    for a in angles:
        Pt = rotate(P, a)
        vec = Pt - B
        L = np.linalg.norm(vec)
        L_act.append(L)
        d_arms.append((Pt[0]*vec[1] - Pt[1]*vec[0]) / (L * 1000.0))
    return np.array(d_arms), np.array(L_act)

d1, _ = get_forces(P1, B1, 0)
d2, _ = get_forces(P2, B2, 0)
M_grav = m * 9.81 * (np.array([rotate(np.array([C_x, C_y]), a)[0] for a in angles])) / 1000.0
# Jednoduchá optimalizace přední síly
F1 = max(50.0, (M_grav[-1] - F2_user * 2 * d2[-1]) / (2 * d1[-1]))
F_user = (M_grav - (F1 * 2 * d1 + F2_user * 2 * d2)) / (1.0) / 9.81

# --- VIZUALIZACE ---
curr = st.slider("Animace", 0.0, float(max_angle), 0.0)
fig, ax = plt.subplots(figsize=(8, 6))
# Víko a těžiště
lid = Polygon(np.array([rotate(pt, curr) for pt in [[0,0], [1000,0], [1000,150], [0,150]]]), facecolor='gray', alpha=0.3)
ax.add_patch(lid)
cg = rotate([C_x, C_y], curr)
ax.plot(*cg, 'ro', markersize=10, label="Těžiště (CG)")
# Vzpěry
for P, B, col in [(P1, B1, 'b'), (P2, B2, 'r')]:
    Pt = rotate(P, curr)
    ax.plot([B[0], Pt[0]], [B[1], Pt[1]], col+'-', lw=3)
    ax.plot(*B, col+'s')
    ax.plot(*Pt, col+'o')

ax.set_aspect('equal')
ax.set_xlim(-200, 1200); ax.set_ylim(-400, 800)
ax.grid(True); ax.legend()
st.pyplot(fig)

st.metric("Síla přední (1ks)", f"{F1:.0f} N")
st.write("Tip: Mrtvý bod zadní vzpěry najdeš v nákresu tak, že zadní červená vzpěra v 45-50° protne bod [0,0] (pant).")
