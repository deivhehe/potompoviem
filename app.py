"""
Návrh geometrie a sil plynových vzpěr pro výklopné víko
=========================================================
Interaktivní Streamlit aplikace pro dimenzování plynových vzpěr (hlavních,
případně + pomocných) výklopného víka (truhla, box, kapota...).

Souřadný systém:
- Počátek [0,0] = osa pantu.
- Lokální souřadnice víka (Lx, Ly) jsou vztaženy k zavřenému stavu (0°),
  kdy Lx směřuje od pantu podél délky víka a Ly je tloušťka/výška víka.
- Úhel theta = úhel otevření (0° = zavřeno, kladný směr = otevírání
  proti směru hodinových ručiček, víko se zvedá od pantu vzhůru).
- Globální pozice libovolného bodu víka: rotace o úhel theta kolem pantu.

Fyzikální model (zjednodušený, ale konzistentní):
- Gravitační moment k pantu: Tg(theta) = -m*g*Xcg(theta)
- Moment od jedné vzpěry (síla F podél osy vzpěra-vana -> vzpěra-víko):
  Ts(theta) = F * (Xb*Yp(theta) - Yb*Xp(theta)) / L(theta)
- Síla do ruky se počítá jako tangenciální síla působící na konci víka
  (rameno = délka víka L_lid): F_ruka(theta) = -(Tg+Ts_celkem)/L_lid
- Mrtvý bod = úhel, kdy Xcg(theta) = 0 (těžiště nad osou pantu).
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq
import time

st.set_page_config(page_title="Návrh plynových vzpěr víka", layout="wide")

G = 9.81  # m/s^2


# ----------------------------------------------------------------------
# Pomocné fyzikální funkce
# ----------------------------------------------------------------------
def rotate(lx, ly, theta):
    """Otočí lokální bod (lx,ly) o úhel theta [rad] kolem pantu [0,0]."""
    c, s = np.cos(theta), np.sin(theta)
    return lx * c - ly * s, lx * s + ly * c


def strut_length(Xb, Yb, lx, ly, theta):
    Xp, Yp = rotate(lx, ly, theta)
    return np.sqrt((Xp - Xb) ** 2 + (Yp - Yb) ** 2)


def signed_moment_arm(Xb, Yb, lx, ly, theta):
    """Znaménkové rameno síly vzpěry vůči pantu (m)."""
    Xp, Yp = rotate(lx, ly, theta)
    L = np.sqrt((Xp - Xb) ** 2 + (Yp - Yb) ** 2)
    if L < 1e-9:
        return 0.0
    return (Xb * Yp - Yb * Xp) / L


def _pick_physical_root(eqs, guesses, L_lid, H_lid, margin=0.15, reject_radius=0.02):
    """Ze všech konvergovaných kořenů vybere ten, který leží v rozumné
    fyzické obálce víka (blízko obdélníku 0..L_lid x 0..H_lid). Pokud
    žádný takový není, vrátí nejlepší dostupný a označí jej jako
    fyzicky nevěrohodný (in_envelope=False)."""
    x_lo, x_hi = -margin * L_lid, (1 + margin) * L_lid
    y_lo, y_hi = -margin * H_lid, (1 + margin) * H_lid

    candidates = []
    for g0 in guesses:
        sol, info, ier, msg = fsolve(eqs, g0, full_output=True)
        res = np.linalg.norm(info["fvec"])
        if res > 1e-6:
            continue
        if np.hypot(*sol) < reject_radius:
            continue  # degenerované řešení v ose pantu
        # dedup podle zaokrouhlení na mm
        key = (round(sol[0], 3), round(sol[1], 3))
        if any(key == c[2] for c in candidates):
            continue
        in_env = (x_lo <= sol[0] <= x_hi) and (y_lo <= sol[1] <= y_hi)
        candidates.append((res, sol, key, in_env))

    if not candidates:
        return None, np.inf, False

    in_env_candidates = [c for c in candidates if c[3]]
    if in_env_candidates:
        best = min(in_env_candidates, key=lambda c: c[0])
        return best[1], best[0], True

    best = min(candidates, key=lambda c: c[0])
    return best[1], best[0], False


def solve_main_pin(Xb, Yb, L0, S, theta_max, L_lid, H_lid):
    """Najde pozici čepu na víku (Lx,Ly) pro hlavní vzpěru tak, aby
    délka @0° = L0 a délka @theta_max = L0+S. Preferuje řešení ležící
    fyzicky na víku."""

    def eqs(v):
        lx, ly = v
        e1 = (lx - Xb) ** 2 + (ly - Yb) ** 2 - L0 ** 2
        Xp2, Yp2 = rotate(lx, ly, theta_max)
        e2 = (Xp2 - Xb) ** 2 + (Yp2 - Yb) ** 2 - (L0 + S) ** 2
        return [e1, e2]

    guesses = [
        (Xb + L0 * 0.6, Yb + L0 * 0.3),
        (Xb - L0 * 0.3, Yb + L0 * 0.6),
        (L0, 0.05),
        (0.05, L0),
        (Xb, Yb + L0),
        (0.3 * L_lid, 0.3 * H_lid),
        (0.6 * L_lid, 0.3 * H_lid),
        (0.7 * L_lid, 0.6 * H_lid),
        (0.2 * L_lid, 0.7 * H_lid),
    ]
    return _pick_physical_root(eqs, guesses, L_lid, H_lid)


def solve_aux_pin(Xb2, Yb2, L02, theta_dead, L_lid, H_lid):
    """Najde pozici čepu na víku pro pomocnou vzpěru: délka @0°=L02 a
    osa vzpěry v theta_dead prochází přesně pantem [0,0].

    Řešeno analyticky: hledaný čep leží na přímce (lx,ly) = t*u, kde
    u = rotate(Xb2,Yb2,-theta_dead). Dosazením do podmínky délky @0°
    vznikne kvadratická rovnice pro t. Řešení existuje pouze pokud
    L02 >= R*sin(theta_dead), kde R = vzdálenost vana-pant (fyzikální
    podmínka proveditelnosti geometrie).

    Vrací: (pozice [Lx,Ly] nebo None, residuum, je_na_víku (bool),
    minimální potřebná zasunutá délka L02 pro proveditelnost).
    """
    R2 = Xb2 ** 2 + Yb2 ** 2
    R = np.sqrt(R2)
    if R < 1e-9:
        return None, np.inf, False, None  # vana v ose pantu - nesmyslné

    sin_d, cos_d = np.sin(theta_dead), np.cos(theta_dead)
    min_L02 = R * abs(sin_d)
    disc = (L02 ** 2) / R2 - sin_d ** 2
    if disc < 0:
        return None, np.inf, False, min_L02  # geometricky neřešitelné

    sq = np.sqrt(disc)
    ux, uy = rotate(Xb2, Yb2, -theta_dead)

    x_lo, x_hi = -0.15 * L_lid, 1.15 * L_lid
    y_lo, y_hi = -0.15 * H_lid, 1.15 * H_lid

    sols = []
    for t in (cos_d + sq, cos_d - sq):
        lx, ly = t * ux, t * uy
        if np.hypot(lx, ly) < 0.02:
            continue  # degenerované řešení v ose pantu
        sols.append((lx, ly))

    if not sols:
        return None, np.inf, False, min_L02

    in_env = [s for s in sols if x_lo <= s[0] <= x_hi and y_lo <= s[1] <= y_hi]
    if in_env:
        return np.array(in_env[0]), 0.0, True, min_L02
    return np.array(sols[0]), 0.0, False, min_L02


def find_dead_point(cg_x, cg_y, theta_max):
    f = lambda th: cg_x * np.cos(th) - cg_y * np.sin(th)
    if f(0.0) * f(theta_max) >= 0:
        return None
    return brentq(f, 1e-6, theta_max)


# ----------------------------------------------------------------------
# UI - Sidebar (vstupy)
# ----------------------------------------------------------------------
st.sidebar.header("1) Geometrie a hmotnost víka")
lid_length = st.sidebar.number_input("Délka víka (mm)", 50.0, 3000.0, 600.0, 10.0)
lid_height = st.sidebar.number_input("Výška / tloušťka víka (mm)", 10.0, 1000.0, 60.0, 5.0)
lid_mass = st.sidebar.number_input("Hmotnost víka (kg)", 0.1, 500.0, 15.0, 0.5)

st.sidebar.header("2) Těžiště víka (od pantu, v zavřeném stavu)")
cg_x_cm = st.sidebar.number_input(
    "Těžiště X (mm)", 0.0, 3000.0, float(np.clip(lid_length * 0.5, 0.0, 3000.0)), 5.0
)
cg_y_cm = st.sidebar.number_input(
    "Těžiště Y (mm)", -500.0, 1000.0, float(np.clip(lid_height * 0.5, -500.0, 1000.0)), 5.0
)

st.sidebar.header("3) Rozsah otevření")
theta_max_deg = st.sidebar.slider("Maximální úhel otevření (°)", 45, 130, 95)

st.sidebar.header("4) Konfigurace vzpěr")
config = st.sidebar.radio("Typ", ["2× hlavní vzpěra", "2× hlavní + 2× pomocná vzpěra"])
use_aux = config.startswith("2× hlavní +")

st.sidebar.subheader("Hlavní vzpěra (1 ks)")
Xb1 = st.sidebar.number_input(
    "Vana X (mm)", -1000.0, 3000.0, float(np.clip(lid_length * 0.35, -1000.0, 3000.0)), 5.0
)
Yb1 = st.sidebar.number_input(
    "Vana Y (mm)", -1000.0, 1000.0, float(np.clip(lid_height * 3.0, -1000.0, 1000.0)), 5.0
)
L0_1 = st.sidebar.number_input("Zasunutá délka @0° (mm)", 30.0, 2000.0, 220.0, 5.0)
S1 = st.sidebar.number_input("Zdvih hlavní vzpěry (mm)", 10.0, 1500.0, 150.0, 5.0)

if use_aux:
    st.sidebar.subheader("Pomocná (zadní) vzpěra (1 ks)")
    Xb2 = st.sidebar.number_input(
        "Vana X pomocná (mm)", -1000.0, 3000.0, float(np.clip(-lid_length * 0.15, -1000.0, 3000.0)), 5.0
    )
    Yb2 = st.sidebar.number_input(
        "Vana Y pomocná (mm)", -1000.0, 1000.0, float(np.clip(lid_height * 2.0, -1000.0, 1000.0)), 5.0
    )
    L0_2 = st.sidebar.number_input("Zasunutá délka pomocné @0° (mm)", 30.0, 2000.0, 150.0, 5.0)
    S2 = st.sidebar.number_input("Zdvih pomocné vzpěry (mm) [info]", 10.0, 1500.0, 100.0, 5.0)

st.sidebar.header("5) Cílové síly do ruky")
target_open_kg = st.sidebar.slider("Síla na otevření @0° (kgf)", 0.5, 15.0, 5.0, 0.5)
target_hold_kg = st.sidebar.slider(
    "Síla do ruky @max. otevření (kgf, záporná = nutno přitáhnout)", -10.0, 10.0, -1.5, 0.5
)

st.sidebar.header("6) Náhled úhlu")
theta_disp_deg = st.sidebar.slider("Úhel pro geometrický náhled (°)", 0, theta_max_deg, 0)
animate = st.sidebar.button("▶️ Animovat otevírání")

# ----------------------------------------------------------------------
# Převody na SI (metry, radiány) - vstupy jsou v milimetrech
# ----------------------------------------------------------------------
mm = 0.001
L_lid = lid_length * mm
H_lid = lid_height * mm
m = lid_mass
cg_x, cg_y = cg_x_cm * mm, cg_y_cm * mm
theta_max = np.radians(theta_max_deg)
n_main = 2

Xb1_m, Yb1_m = Xb1 * mm, Yb1 * mm
L0_1_m, S1_m = L0_1 * mm, S1 * mm

pin1, res1, in_env1 = solve_main_pin(Xb1_m, Yb1_m, L0_1_m, S1_m, theta_max, L_lid, H_lid)
lx1, ly1 = pin1

theta_dead = find_dead_point(cg_x, cg_y, theta_max)

pin2 = None
if use_aux:
    Xb2_m, Yb2_m = Xb2 * mm, Yb2 * mm
    L0_2_m = L0_2 * mm
    if theta_dead is None:
        st.warning(
            "Těžiště víka nepřechází v zadaném rozsahu úhlů přes osu pantu "
            "(žádný mrtvý bod). Pozici čepu pomocné vzpěry nelze exaktně "
            "dopočítat dle zadané podmínky – zkontrolujte polohu těžiště "
            "nebo max. úhel otevření."
        )
    else:
        pin2, res2, in_env2, min_L02 = solve_aux_pin(Xb2_m, Yb2_m, L0_2_m, theta_dead, L_lid, H_lid)
        if pin2 is None:
            R_vana = np.hypot(Xb2_m, Yb2_m)
            st.error(
                f"❌ Zadaná geometrie pomocné vzpěry nemá řešení: vana je od pantu vzdálená "
                f"{R_vana/mm:.0f} mm, ale zasunutá délka @0° je jen {L0_2:.0f} mm. "
                f"Aby úloha měla řešení, musí platit L₀ ≥ {min_L02/mm:.0f} mm "
                f"(vzdálenost vana–pant × sin(úhel mrtvého bodu)). "
                f"Buď zvětši zasunutou délku pomocné vzpěry alespoň na ~{min_L02/mm:.0f} mm, "
                f"nebo posuň vanu pomocné vzpěry blíž k pantu."
            )
            res2, in_env2 = np.inf, False
        else:
            lx2, ly2 = pin2
            n_aux = 2

# ----------------------------------------------------------------------
# Výpočet potřebných sil vzpěr (kvazistatická rovnováha momentů)
# ----------------------------------------------------------------------
def Xcg(theta):
    return cg_x * np.cos(theta) - cg_y * np.sin(theta)


def Tg(theta):
    return -m * G * Xcg(theta)


target_open_N = target_open_kg * G
target_hold_N = target_hold_kg * G

d1_0 = signed_moment_arm(Xb1_m, Yb1_m, lx1, ly1, 0.0)
d1_max = signed_moment_arm(Xb1_m, Yb1_m, lx1, ly1, theta_max)

if use_aux and pin2 is not None:
    d2_0 = signed_moment_arm(Xb2_m, Yb2_m, lx2, ly2, 0.0)
    d2_max = signed_moment_arm(Xb2_m, Yb2_m, lx2, ly2, theta_max)

    # soustava:  n1*d1*F1 + n2*d2*F2 = -(Tg + F_hand_target*L_lid)
    A = np.array([[n_main * d1_0, n_aux * d2_0], [n_main * d1_max, n_aux * d2_max]])
    b = np.array(
        [-(Tg(0.0) + target_open_N * L_lid), -(Tg(theta_max) + target_hold_N * L_lid)]
    )
    try:
        F_main, F_aux = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        F_main, F_aux = 0.0, 0.0
        st.error("Soustavu pro výpočet sil se nepodařilo vyřešit (singulární geometrie).")
else:
    F_aux = None
    denom = n_main * d1_0
    F_main = (-(Tg(0.0) + target_open_N * L_lid)) / denom if abs(denom) > 1e-9 else 0.0


def F_hand(theta):
    """Síla do ruky (N), tangenciálně na konci víka."""
    Ts = n_main * F_main * signed_moment_arm(Xb1_m, Yb1_m, lx1, ly1, theta)
    if use_aux and pin2 is not None and F_aux is not None:
        Ts += n_aux * F_aux * signed_moment_arm(Xb2_m, Yb2_m, lx2, ly2, theta)
    return -(Tg(theta) + Ts) / L_lid


# ----------------------------------------------------------------------
# Metrický panel
# ----------------------------------------------------------------------
st.title("🔧 Návrh plynových vzpěr výklopného víka")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Síla hlavní vzpěry (1 ks)", f"{F_main:.0f} N", f"{F_main/G:.1f} kgf")
if use_aux and F_aux is not None:
    c2.metric("Síla pomocné vzpěry (1 ks)", f"{F_aux:.0f} N", f"{F_aux/G:.1f} kgf")
else:
    c2.metric("Síla pomocné vzpěry", "—")
c3.metric("Síla do ruky @0°", f"{F_hand(0.0)/G:.2f} kgf")
c4.metric(
    "Síla do ruky @max",
    f"{F_hand(theta_max)/G:.2f} kgf",
    "pomáhá zavírat" if F_hand(theta_max) < 0 else "ještě otevírá",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Čep na víku – hlavní X", f"{lx1/mm:.1f} mm")
c6.metric("Čep na víku – hlavní Y", f"{ly1/mm:.1f} mm")
if use_aux and pin2 is not None:
    c7.metric("Čep na víku – pomocná X", f"{lx2/mm:.1f} mm")
    c8.metric("Čep na víku – pomocná Y", f"{ly2/mm:.1f} mm")
else:
    c7.metric("Čep na víku – pomocná X", "—")
    c8.metric("Čep na víku – pomocná Y", "—")

if theta_dead is not None:
    st.info(f"🔹 Mrtvý bod (těžiště nad pantem) při úhlu **{np.degrees(theta_dead):.1f}°**")
else:
    st.info("🔹 Mrtvý bod nebyl v zadaném rozsahu úhlů nalezen.")

if pin1 is None or res1 > 1e-6:
    st.warning("Pozor: pro hlavní vzpěru se nepodařilo najít platné řešení – zkontrolujte vstupy (vana X/Y, zasunutá délka, zdvih).")
elif not in_env1:
    st.warning(
        f"⚠️ Čep hlavní vzpěry na víku vychází **mimo těleso víka** "
        f"(X={lx1/mm:.0f} mm, Y={ly1/mm:.0f} mm, rozměry víka {lid_length:.0f}×{lid_height:.0f} mm). "
        "To znamená, že zadaná zasunutá délka/zdvih/pozice vany nejsou s reálnou geometrií víka slučitelné. "
        "Zkus upravit pozici vany (X, Y) nebo zasunutou délku a zdvih hlavní vzpěry."
    )

if use_aux and pin2 is not None:
    if res2 > 1e-6:
        st.warning("Pozor: pro pomocnou vzpěru se nepodařilo najít platné řešení – zkontrolujte vstupy.")
    elif not in_env2:
        st.warning(
            f"⚠️ Čep pomocné vzpěry na víku vychází **mimo těleso víka** "
            f"(X={lx2/mm:.0f} mm, Y={ly2/mm:.0f} mm, rozměry víka {lid_length:.0f}×{lid_height:.0f} mm). "
            "Podmínka „osa vzpěry prochází pantem v mrtvém bodě“ tak s aktuální pozicí vany a zasunutou "
            "délkou nemá fyzicky rozumné řešení na víku. Zkus posunout vanu pomocné vzpěry (X, Y) blíž "
            "k pantu nebo upravit zasunutou délku @0°."
        )

st.divider()


# ----------------------------------------------------------------------
# Vykreslení geometrie
# ----------------------------------------------------------------------
def draw_geometry(ax, theta):
    ax.clear()
    # obrys "vany" / boxu - jen orientační referenční obdélník
    box_w = max(L_lid, abs(Xb1_m) + 0.1, abs(Xb2_m) + 0.1 if use_aux else 0)
    ax.add_patch(
        plt.Rectangle((-0.02, -H_lid * 4), box_w + 0.02, H_lid * 4, fill=False,
                      edgecolor="gray", linestyle=":", linewidth=1)
    )

    # víko jako obdélník (4 rohy v lokálních souřadnicích -> rotace)
    corners_local = [(0, 0), (L_lid, 0), (L_lid, H_lid), (0, H_lid)]
    corners_global = [rotate(lx, ly, theta) for lx, ly in corners_local]
    xs = [p[0] for p in corners_global] + [corners_global[0][0]]
    ys = [p[1] for p in corners_global] + [corners_global[0][1]]
    ax.fill(xs, ys, color="#c9a876", alpha=0.6, edgecolor="black", linewidth=1.5, zorder=3)

    # pant
    ax.plot(0, 0, "ko", markersize=8, zorder=5)
    ax.annotate("Pant", (0, 0), textcoords="offset points", xytext=(-8, -12))

    # těžiště
    Xc, Yc = rotate(cg_x, cg_y, theta)
    ax.plot(Xc, Yc, "o", color="red", markersize=10, zorder=6)
    ax.annotate("CG", (Xc, Yc), textcoords="offset points", xytext=(6, 6), color="red")

    # hlavní vzpěra
    Xp1, Yp1 = rotate(lx1, ly1, theta)
    ax.plot([Xb1_m, Xp1], [Yb1_m, Yp1], "-", color="#1f77b4", linewidth=3, zorder=4, label="Hlavní vzpěra")
    ax.plot(Xb1_m, Yb1_m, "s", color="#1f77b4", markersize=7, zorder=5)
    ax.plot(Xp1, Yp1, "^", color="#1f77b4", markersize=7, zorder=5)

    # pomocná vzpěra
    if use_aux and pin2 is not None:
        Xp2, Yp2 = rotate(lx2, ly2, theta)
        ax.plot([Xb2_m, Xp2], [Yb2_m, Yp2], "-", color="#d62728", linewidth=3, zorder=4, label="Pomocná vzpěra")
        ax.plot(Xb2_m, Yb2_m, "s", color="#d62728", markersize=7, zorder=5)
        ax.plot(Xp2, Yp2, "^", color="#d62728", markersize=7, zorder=5)

    lim = max(L_lid, box_w) * 1.3 + 0.05
    ax.set_xlim(-lim * 0.5, lim)
    ax.set_ylim(-H_lid * 4 - 0.05, lim)
    ax.set_aspect("equal")
    ax.invert_xaxis()  # pant vpravo dole, X roste doleva (dle reálné orientace)
    ax.set_title(f"Geometrie víka @ {np.degrees(theta):.1f}°")
    ax.set_xlabel("X (m) — kladně doleva od pantu")
    ax.set_ylabel("Y (m) — kladně nahoru od pantu")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def draw_force_profile(ax, theta_marker=None):
    ax.clear()
    thetas = np.linspace(0, theta_max, 200)
    forces_kg = np.array([F_hand(t) / G for t in thetas])
    degs = np.degrees(thetas)

    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between(degs, forces_kg, 0, where=(forces_kg >= 0), color="#ff7f0e", alpha=0.5, label="Nutno tlačit (otevírání)")
    ax.fill_between(degs, forces_kg, 0, where=(forces_kg < 0), color="#2ca02c", alpha=0.5, label="Vzpěra pomáhá / brzdí")
    ax.plot(degs, forces_kg, color="black", linewidth=1.5)

    if theta_dead is not None:
        ax.axvline(np.degrees(theta_dead), color="purple", linestyle="--", linewidth=1.5, label="Mrtvý bod")

    if theta_marker is not None:
        ax.plot(np.degrees(theta_marker), F_hand(theta_marker) / G, "o", color="black", markersize=9, zorder=6)

    ax.set_xlabel("Úhel otevření (°)")
    ax.set_ylabel("Síla do ruky (kgf)")
    ax.set_title("Profil síly do ruky vs. úhel otevření")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)


col_geo, col_force = st.columns(2)
fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))

theta_disp = np.radians(theta_disp_deg)

if animate:
    placeholder1 = col_geo.empty()
    placeholder2 = col_force.empty()
    for deg in np.linspace(0, theta_max_deg, 40):
        th = np.radians(deg)
        draw_geometry(ax1, th)
        draw_force_profile(ax2, th)
        placeholder1.pyplot(fig1)
        placeholder2.pyplot(fig2)
        time.sleep(0.04)
else:
    draw_geometry(ax1, theta_disp)
    draw_force_profile(ax2, theta_disp)
    col_geo.pyplot(fig1)
    col_force.pyplot(fig2)

st.caption(
    "Model počítá kvazistaticky s momentovou rovnováhou k ose pantu. "
    "Síla do ruky je odvozena jako tangenciální síla na konci víka "
    "(rameno = délka víka). Reálné plynové vzpěry mají mírně proměnnou "
    "sílu podle vysunutí – pro přesné dimenzování vždy ověřte u výrobce."
)
