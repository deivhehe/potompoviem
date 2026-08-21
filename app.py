import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve, brentq
import time

st.set_page_config(page_title="Návrh plynových vzpěr víka", layout="wide")

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
    if L_mm < 1e-6: return 0.0
    return (Xb_mm * Yp_mm - Yb_mm * Xp_mm) / (L_mm * 1000.0)

def _pick_physical_root(eqs, guesses, L_lid_mm, H_lid_mm, margin=0.25, reject_radius=2.0):
    candidates = []
    for g0 in guesses:
        sol, info, ier, msg = fsolve(eqs, g0, full_output=True)
        res = np.linalg.norm(info["fvec"])
        if res > 1e-3 or np.hypot(*sol) < reject_radius: continue
        candidates.append((res, sol))
    if not candidates: return None, np.inf, False
    best = min(candidates, key=lambda c: c[0])
    return best[1], best[0], True

def solve_main_pin_mm(Xb_mm, Yb_mm, L0_mm, S_mm, theta_max, L_lid_mm, H_lid_mm):
    def eqs(v):
        lx, ly = v
        e1 = (lx - Xb_mm)**2 + (ly - Yb_mm)**2 - L0_mm**2
        Xp2, Yp2 = rotate_mm(lx, ly, theta_max)
        e2 = (Xp2 - Xb_mm)**2 + (Yp2 - Yb_mm)**2 - (L0_mm + S_mm)**2
        return [e1, e2]
    guesses = [(Xb_mm + L0_mm*0.5, Yb_mm + L0_mm*0.5), (Xb_mm - L0_mm*0.2, Yb_mm + L0_mm*0.8)]
    return _pick_physical_root(eqs, guesses, L_lid_mm, H_lid_mm)

def solve_aux_pin_mm(Xb2_mm, Yb2_mm, L02_mm, theta_dead, L_lid_mm, H_lid_mm):
    R2 = Xb2_mm**2 + Yb2_mm**2
    R = np.sqrt(R2)
    if R < 1e-6: return None, np.inf, False, None
    sin_d, cos_d = np.sin(theta_dead), np.cos(theta_dead)
    disc = (L02_mm**2) / R2 - sin_d**2
    if disc < 0: return None, np.inf, False, None
    sq = np.sqrt(disc)
    ux, uy = rotate_mm(Xb2_mm, Yb2_mm, -theta_dead)
    return np.array([(cos_d + sq)*ux, (cos_d + sq)*uy]), 0.0, True, 0.0

def find_dead_point(cg_x_mm, cg_y_mm, theta_max):
    f = lambda th: cg_x_mm * np.cos(th) - cg_y_mm * np.sin(th)
    if f(0.0) * f(theta_max) >= 0: return None
    return brentq(f, 1e-6, theta_max)

# ----------------------------------------------------------------------
# UI - Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("Parametry víka a vzpěr")
lid_length = st.sidebar.number_input("Délka víka (mm)", 1109.0)
lid_height = st.sidebar.number_input("Výška víka (mm)", 812.0)
lid_mass = st.sidebar.number_input("Hmotnost víka (kg)", 180.0)
theta_max_deg = st.sidebar.slider("Maximální úhel (°)", 45, 130, 83)

st.sidebar.subheader("Hlavní vzpěra")
Xb1 = st.sidebar.number_input("Vana X hl. (mm)", 585.0)
Yb1 = st.sidebar.number_input("Vana Y hl. (mm)", -111.0)
L0_1 = st.sidebar.number_input("Zasunutá délka hl. (mm)", 618.0)
S1 = st.sidebar.number_input("Zdvih hl. (mm)", 500.0)

st.sidebar.subheader("Pomocná vzpěra")
Xb2 = st.sidebar.number_input("Vana X pomocná (mm)", 145.0)
Yb2 = st.sidebar.number_input("Vana Y pomocná (mm)", -241.0)
L0_2 = st.sidebar.number_input("Zasunutá délka pom. (mm)", 561.0)
F_aux_catalog = st.sidebar.number_input("Síla pomocné (N)", 500.0)

# ----------------------------------------------------------------------
# Výpočty
# ----------------------------------------------------------------------
theta_max = np.radians(theta_max_deg)
n_main = 2
n_aux = 2

pin1, _, _ = solve_main_pin_mm(Xb1, Yb1, L0_1, S1, theta_max, lid_length, lid_height)
if pin1 is None: st.error("Chyba výpočtu geometrie."); st.stop()
lx1, ly1 = pin1

theta_dead = find_dead_point(lid_length/2, lid_height/2, theta_max) # Zjednodušený odhad mrtvého bodu
pin2, _, _, _ = solve_aux_pin_mm(Xb2, Yb2, L0_2, theta_dead if theta_dead else 0.5, lid_length, lid_height)
lx2, ly2 = pin2 if pin2 is not None else (0,0)

def Tg(theta): return -lid_mass * G * (lid_length/2000 * np.cos(theta))
d1_0 = signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, 0.0)
d2_0 = signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, 0.0)

F_main = (-(Tg(0.0) + 50.0 * (lid_length/1000)) - (n_aux * d2_0 * F_aux_catalog)) / (n_main * d1_0)

def F_hand(theta):
    # Fáze: Hlavní tlačí, pomocná v závislosti na úhlu (přes mrtvý bod) pomáhá nebo brzdí
    Ts_main = n_main * F_main * signed_moment_arm_mm(Xb1, Yb1, lx1, ly1, theta)
    Ts_aux = n_aux * F_aux_catalog * signed_moment_arm_mm(Xb2, Yb2, lx2, ly2, theta)
    return -(Tg(theta) + Ts_main + Ts_aux) / (lid_length/1000)

# ----------------------------------------------------------------------
# UI - Výsledky
# ----------------------------------------------------------------------
st.title("🔧 Návrh plynových vzpěr")
c1, c2 = st.columns(2)
c1.metric("Síla hlavní vzpěry", f"{F_main:.0f} N")
c2.metric("Síla do ruky @0°", f"{F_hand(0.0):.1f} N")

# Vykreslení grafu
fig, ax = plt.subplots(figsize=(6, 4))
thetas = np.linspace(0, theta_max, 100)
forces = [F_hand(t) for t in thetas]
ax.plot(np.degrees(thetas), forces, label="Síla na ruku")
ax.axhline(0, color='red', linestyle='--')
ax.set_title("Profil síly (kladná = člověk tlačí, záporná = vzpěry brzdí/přetlačují)")
ax.grid(True)
st.pyplot(fig)

st.info("Logika výpočtu: Fáze 0-42.5°: Obě vzpěry pomáhají. Fáze 42.5-83°: Pomocná vzpěra po překročení mrtvého bodu působí proti hlavní vzpěře (brzdí).")
