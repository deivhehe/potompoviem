import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

st.set_page_config(layout="wide", page_title="Kalkulačka vzpěr PRO v2")
st.title("Profi kalkulačka plynových vzpěr (3D víko a vlastní rozměry)")

# --- UŽIVATELSKÉ ROZHRANÍ (Postranní panel) ---
st.sidebar.header("1. Parametry víka (krabice)")
m = st.sidebar.number_input("Hmotnost víka (kg)", value=25.0, step=1.0)
L_lid = st.sidebar.number_input("Délka víka (mm)", value=1000.0, step=10.0)
H_lid = st.sidebar.number_input("Výška/Tloušťka víka (mm)", value=150.0, step=10.0)

st.sidebar.subheader("Těžiště víka (v zavřeném stavu)")
C_x = st.sidebar.number_input("Těžiště osa X (mm od pantu)", value=L_lid/2, step=10.0)
C_y = st.sidebar.number_input("Těžiště osa Y (mm od pantu)", value=H_lid/2, step=10.0)
C_0 = np.array([C_x, C_y]) 

max_angle = st.sidebar.slider("Požadovaný úhel otevření (°)", 45, 110, 80)

st.sidebar.header("2. Pozice na vaně (pevný čep vzpěry)")
st.sidebar.info("Referenční bod [0,0] je osa pantu. Čep na vaně je typicky X > 0, Y < 0.")
B_x = st.sidebar.number_input("Čep vana X (mm)", value=150.0, step=10.0)
B_y = st.sidebar.number_input("Čep vana Y (mm)", value=-200.0, step=10.0)
B = np.array([B_x, B_y])

st.sidebar.header("3. Rozměry vzpěry")
L_ext = st.sidebar.number_input("Celková délka - vysunutá (mm)", value=500.0, step=10.0, help="Od středu čepu po střed čepu")
stroke = st.sidebar.number_input("Zdvih vzpěry (mm)", value=200.0, step=10.0)
pocet_vzper = st.sidebar.radio("Počet vzpěr celkem", [2, 4], index=0)

# Výpočet zasunuté délky
L_comp = L_ext - stroke
if L_comp <= stroke:
    st.sidebar.warning("⚠️ Fyzikální varování: Zasunutá délka těla vzpěry je menší než zdvih. Trubka vzpěry by nedokázala pojmout pístnici.")

g = 9.81
H = np.array([0.0, 0.0]) # Pant

# --- MATEMATIKA A KINEMATIKA ---
def rotate(pt, origin, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        origin[0] + (pt[0] - origin[0]) * np.cos(a) - (pt[1] - origin[1]) * np.sin(a),
        origin[1] + (pt[0] - origin[0]) * np.sin(a) + (pt[1] - origin[1]) * np.cos(a)
    ])

def find_lid_mount(B, L_comp, L_ext, max_angle, H):
    # Hledání průsečíku dvou kružnic
    B_rot = rotate(B, H, -max_angle)
    d = np.linalg.norm(B_rot - B)
    if d > (L_comp + L_ext) or d < abs(L_comp - L_ext) or d == 0:
        return None # Nelze sestrojit
    
    a = (L_comp**2 - L_ext**2 + d**2) / (2 * d)
    val = L_comp**2 - a**2
    if val < 0: return None
    h = np.sqrt(val)
    P2 = B + a * (B_rot - B) / d
    
    x3 = P2[0] + h * (B_rot[1] - B[1]) / d
    y3 = P2[1] - h * (B_rot[0] - B[0]) / d
    x4 = P2[0] - h * (B_rot[1] - B[1]) / d
    y4 = P2[1] + h * (B_rot[0] - B[0]) / d
    
    p3, p4 = np.array([x3, y3]), np.array([x4, y4])
    
    # Pro víko hledáme bod v rozumném kvadrantu (X > 0)
    if p3[0] > 0 and p4[0] <= 0: return p3
    if p4[0] > 0 and p3[0] <= 0: return p4
    return p3 if p3[1] > p4[1] else p4 # Preferujeme bod výše

# 1. Nalezení bodu na víku
P0 = find_lid_mount(B, L_comp, L_ext, max_angle, H)

if P0 is None:
    st.error("❌ Pro tuto kombinaci pozice čepu na vaně, úhlu a délek vzpěry NEEXISTUJE geometrické řešení. Vzpěra je příliš krátká nebo naopak dlouhá, případně je čep na vaně moc blízko pantu.")
else:
    # 2. Kinematika a výpočet sil v celém rozsahu
    angles = np.linspace(0, max_angle, 100)
    M_grav = m * g * (np.array([rotate(C_0, H, a)[0] for a in angles]) - H[0]) / 1000.0 # Nm
    
    d_arms = []
    L_actual = []
    for alpha in angles:
        P_t = rotate(P0, H, alpha)
        vec = P_t - B
        L = np.linalg.norm(vec)
        L_actual.append(L)
        u = vec / L
        r = P_t - H
        d_arms.append((r[0]*u[1] - r[1]*u[0]) / 1000.0)
    
    d_arms = np.array(d_arms)
    L_actual = np.array(L_actual)
    
    # Výpočet síly pro překonání gravitace
    valid_idx = np.abs(d_arms) > 0.01
    if np.any(valid_idx):
        required_F_total = np.max((M_grav[valid_idx] * 1.1) / np.abs(d_arms[valid_idx]))
    else:
        required_F_total = 0

    F_1_strut = required_F_total / pocet_vzper
    F_1_strut_rounded = np.ceil(F_1_strut / 50.0) * 50 # Zaokrouhlení na nejbližších 50 N

    # Uživatelská síla na hraně víka (v kg)
    M_struts_actual = (F_1_strut_rounded * pocet_vzper) * d_arms
    M_net = M_struts_actual - M_grav
    F_user_kg = (M_net / (L_lid / 1000.0)) / g

    # --- ZOBRAZENÍ VÝSLEDKŮ ---
    st.success(f"✅ Geometrie nalezena! Čep na víku vyvrtat ve vzdálenosti X: **{P0[0]:.1f} mm**, Y: **{P0[1]:.1f} mm** od pantu.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zasunutá délka vzpěry", f"{L_comp:.1f} mm", "Dopočítáno")
    col2.metric("Potřebná síla vzpěry", f"{F_1_strut_rounded:.0f} N", f"(1 vzpěra z {pocet_vzper})")
    col3.metric("Síla na začátku (0°)", f"{-F_user_kg[0]:.1f} kg", "Kladné = těžké do ruky")
    col4.metric("Síla na konci (otevřeno)", f"{-F_user_kg[-1]:.1f} kg", "Záporné = drží samo")

    st.divider()

    # --- INTERAKTIVNÍ VIZUALIZACE ---
    st.subheader("Vizualizace a dráha víka")
    current_angle = st.slider("🔍 Aktuální úhel víka pro zobrazení (Animace)", 0.0, float(max_angle), 0.0, step=1.0)
    
    idx = int((current_angle / max_angle) * 99)
    cur_user_F = -F_user_kg[idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graf 1: Nákres 2D Profilu
    ax1.plot(*H, 'ko', markersize=8, label='Pant [0,0]', zorder=5)
    ax1.plot(*B, 'bs', markersize=8, label='Čep na vaně', zorder=5)
    
    # Vykreslení víka jako polygonu
    lid_poly = np.array([
        [0, 0],
        [L_lid, 0],
        [L_lid, H_lid],
        [0, H_lid]
    ])
    lid_poly_rotated = np.array([rotate(pt, H, current_angle) for pt in lid_poly])
    polygon = Polygon(lid_poly_rotated, closed=True, fill=True, facecolor='gray', alpha=0.3, edgecolor='black', lw=2)
    ax1.add_patch(polygon)

    # Těžiště a čep na víku v aktuálním úhlu
    C_cur = rotate(C_0, H, current_angle)
    P_cur = rotate(P0, H, current_angle)
    
    ax1.plot(*C_cur, 'gx', markersize=8, label='Těžiště')
    ax1.plot([B[0], P_cur[0]], [B[1], P_cur[1]], 'r-', lw=3, label=f'Vzpěra ({L_actual[idx]:.0f} mm)')
    ax1.plot(*P_cur, 'ro', markersize=6)

    # Vizuální limity a úpravy grafu 1
    ax1.set_aspect('equal')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title(f"2D Průřez při úhlu {current_angle:.1f}°")
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=':')

    # Dynamické přizpůsobení os, aby graf neposkakoval při animaci
    max_radius = max(L_lid, abs(B_x), abs(B_y)) * 1.2
    ax1.set_xlim(-max_radius*0.2, max_radius)
    ax1.set_ylim(-max_radius*0.5, max_radius*1.1)

    # Graf 2: Síla do ruky vs Úhel
    ax2.plot(angles, -F_user_kg, 'b-', lw=2)
    ax2.axhline(0, color='black', lw=1)
    ax2.plot(current_angle, cur_user_F, 'ro', markersize=10, label=f"Nyní: {cur_user_F:.1f} kg")
    
    ax2.fill_between(angles, 0, -F_user_kg, where=(-F_user_kg >= 0), facecolor='red', alpha=0.2, label="Víko padá (musíš zvedat)")
    ax2.fill_between(angles, 0, -F_user_kg, where=(-F_user_kg < 0), facecolor='green', alpha=0.2, label="Víko se otevírá (drží samo)")

    ax2.set_xlabel("Úhel otevření víka (°)")
    ax2.set_ylabel("Síla potřebná na okraji víka (kg)")
    ax2.set_title("Profil síly uživatele")
    ax2.legend()
    ax2.grid(True, linestyle=':')

    st.pyplot(fig)
