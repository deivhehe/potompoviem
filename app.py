import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Vzpěrovač PRO")

# --- UI ---
st.title("Vzpěrovač - Profesionální návrh")
st.sidebar.header("Parametry")
m = st.sidebar.number_input("Hmotnost víka (kg)", 30.0)
L_lid = st.sidebar.number_input("Délka víka (mm)", 1000.0)
H_lid = st.sidebar.number_input("Výška víka (mm)", 150.0)
C = np.array([st.sidebar.number_input("Těžiště X (mm)", 500.0), st.sidebar.number_input("Těžiště Y (mm)", 75.0)])
max_angle = st.sidebar.slider("Max. úhel otevření (°)", 45, 110, 80)

st.sidebar.header("Hlavní vzpěra")
B1 = np.array([st.sidebar.number_input("Vana X1", 520.0), st.sidebar.number_input("Vana Y1", -126.0)])
L_c1 = st.sidebar.number_input("Zasunutá délka 1 (mm)", 618.0)
S1 = st.sidebar.number_input("Zdvih 1 (mm)", 500.0)

st.sidebar.header("Pomocná vzpěra")
B2 = np.array([st.sidebar.number_input("Vana X2", 175.0), st.sidebar.number_input("Vana Y2", -301.0)])
L_c2 = st.sidebar.number_input("Zasunutá délka 2 (mm)", 560.0)
S2 = st.sidebar.number_input("Zdvih 2 (mm)", 120.0)

# --- MATEMATIKA ---
def get_P(B, L_closed, stroke, angle_open):
    L_open = L_closed + stroke
    # Průsečík kružnic: S1=B, r1=L_c; S2=B_rot, r2=L_o
    B_rot = np.array([B[0]*np.cos(np.radians(-angle_open)) - B[1]*np.sin(np.radians(-angle_open)), 
                      B[0]*np.sin(np.radians(-angle_open)) + B[1]*np.cos(np.radians(-angle_open))])
    d = np.linalg.norm(B_rot - B)
    a = (L_closed**2 - L_open**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, L_closed**2 - a**2))
    P2 = B + a * (B_rot - B) / d
    # Dvě řešení, bereme to, co dává smysl pro víko (kladné Y)
    return np.array([P2[0] - h * (B_rot[1] - B[1]) / d, P2[1] + h * (B_rot[0] - B[0]) / d])

P1 = get_P(B1, L_c1, S1, max_angle)
P2 = get_P(B2, L_c2, S2, max_angle)

# Výpočet sil
angles = np.linspace(0, max_angle, 100)
M_grav = m * 9.81 * (np.array([C[0]*np.cos(np.radians(a)) - C[1]*np.sin(np.radians(a)) for a in angles])) / 1000.0

def get_d(P, B, a):
    Pt = np.array([P[0]*np.cos(np.radians(a)) - P[1]*np.sin(a), P[0]*np.sin(a) + P[1]*np.cos(a)])
    vec = Pt - B
    L = np.linalg.norm(vec)
    return (Pt[0]*vec[1] - Pt[1]*vec[0]) / (L * 1000.0)

d1 = np.array([get_d(P1, B1, a) for a in angles])
d2 = np.array([get_d(P2, B2, a) for a in angles])

# F1: Aby v 0° síla na zvednutí byla 5 kg
F1 = (M_grav[0] - 5*9.81*L_lid/1000 - F2_user*2*d2[0]) / (2*d1[0])
F1 = max(50.0, F1)
F_user = (M_grav - (F1*2*d1 + F2_user*2*d2)) / (L_lid/1000) / 9.81

# --- VÝSTUP ---
col1, col2, col3 = st.columns(3)
col1.metric("Pozice čepu víko 1 [X, Y]", f"[{P1[0]:.0f}, {P1[1]:.0f}]")
col2.metric("Pozice čepu víko 2 [X, Y]", f"[{P2[0]:.0f}, {P2[1]:.0f}]")
col3.metric("Síla hlavní vzpěry (1ks)", f"{F1:.0f} N")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# Vizualizace
curr = st.slider("Animace", 0.0, float(max_angle), 0.0)
ax1.plot(*rotate(C, curr), 'ro', markersize=10, label="CG")
for P, B, col in [(P1, B1, 'b'), (P2, B2, 'r')]:
    Pt = rotate(P, curr)
    ax1.plot([B[0], Pt[0]], [B[1], Pt[1]], col+'-')
ax1.set_xlim(-200, 1200); ax1.set_ylim(-400, 800); ax1.grid()
ax2.plot(angles, F_user); ax2.axhline(0, color='k'); ax2.grid()
ax2.set_title("Síla do ruky (kg)")
st.pyplot(fig)
