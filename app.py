import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# PLYNOVÉ VZPĚRY – NÁVRH GEOMETRIE VÍKA
# ============================================================

st.set_page_config(
    page_title="Návrh plynových vzpěr",
    layout="wide",
)

st.title("Návrh a kontrola geometrie plynových vzpěr")
st.caption(
    "2D statický model výklopného víka. "
    "Úhel 0° = zavřeno, kladný směr = otevírání."
)


# ============================================================
# POMOCNÉ FUNKCE
# ============================================================

def rotate_point(point, angle):
    """Rotace bodu kolem pantu [0,0]."""
    c = math.cos(angle)
    s = math.sin(angle)

    x, y = point

    return np.array([
        c * x - s * y,
        s * x + c * y
    ])


def distance(a, b):
    return float(np.linalg.norm(a - b))


def cross_2d(r, f):
    """Moment síly v 2D."""
    return r[0] * f[1] - r[1] * f[0]


def spring_torque(base, lid_point, force):
    """
    Moment plynové vzpěry kolem pantu.

    Vzpěra tlačí od bodu na víku směrem k bodu na vaně.
    """
    vector = base - lid_point
    length = np.linalg.norm(vector)

    if length < 1e-9:
        return 0.0

    force_vector = force * vector / length

    return cross_2d(lid_point, force_vector)


def gravity_torque(cg, angle, mass):
    """
    Moment gravitace kolem pantu.
    """
    cg_world = rotate_point(cg, angle)

    gravity = np.array([
        0.0,
        -mass * 9.81
    ])

    return cross_2d(cg_world, gravity)


def solve_two_circles(c1, r1, c2, r2):
    """
    Průsečík dvou kružnic.

    Používá se pro přesný výpočet čepu hlavní vzpěry.
    """

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]

    d = math.sqrt(dx * dx + dy * dy)

    if d < 1e-9:
        return []

    if d > r1 + r2:
        return []

    if d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)

    h_squared = r1**2 - a**2

    if h_squared < 0:
        h_squared = 0

    h = math.sqrt(h_squared)

    xm = c1[0] + a * dx / d
    ym = c1[1] + a * dy / d

    rx = -dy * h / d
    ry = dx * h / d

    p1 = np.array([
        xm + rx,
        ym + ry
    ])

    p2 = np.array([
        xm - rx,
        ym - ry
    ])

    return [p1, p2]


def solve_main_pin(base, closed_length, stroke, max_angle):
    """
    Výpočet čepu hlavní vzpěry na víku.

    V zavřeném stavu:
        |P - base| = closed_length

    V otevřeném stavu:
        |R(P) - base| = closed_length + stroke

    Po převedení otevřeného stavu zpět do 0° soustavy
    vznikne průsečík dvou kružnic.
    """

    target_length = closed_length + stroke

    # Bod základny převedený do soustavy víka při 0°
    base_rotated_back = rotate_point(
        base,
        -max_angle
    )

    candidates = solve_two_circles(
        base,
        closed_length,
        base_rotated_back,
        target_length
    )

    return candidates


def solve_aux_pin(base, closed_length, dead_angle):
    """
    Pomocná vzpěra.

    Podmínka:
    v mrtvém bodě musí osa vzpěry procházet přes pant.

    Proto hledáme bod Q na přímce:
        base -> pant

    ve vzdálenosti closed_length od základny.

    Q je pozice čepu na víku v mrtvém bodě.
    Následně jej otočíme zpět do 0°.
    """

    direction = -base

    d = np.linalg.norm(direction)

    if d < 1e-9:
        return []

    direction = direction / d

    # Dva možné body na přímce.
    q1 = base + direction * closed_length
    q2 = base - direction * closed_length

    p1 = rotate_point(
        q1,
        -dead_angle
    )

    p2 = rotate_point(
        q2,
        -dead_angle
    )

    return [p1, p2]


def calculate_hand_force(
    angle,
    cg,
    mass,
    springs,
    hand_point
):
    """
    Výpočet síly do ruky.

    Síla do ruky je uvažována jako tečná síla.
    """

    hand_world = rotate_point(
        hand_point,
        angle
    )

    hand_radius = np.linalg.norm(
        hand_world
    )

    if hand_radius < 1e-9:
        return np.nan

    total_torque = gravity_torque(
        cg,
        angle,
        mass
    )

    for spring in springs:

        base = spring["base"]
        pin = spring["pin"]
        force = spring["force"]

        pin_world = rotate_point(
            pin,
            angle
        )

        total_torque += spring_torque(
            base,
            pin_world,
            force
        )

    # Tangenciální síla:
    # M = F * r
    return -total_torque / hand_radius


def calculate_profile(
    angles,
    cg,
    mass,
    springs,
    hand_point
):

    result = []

    for angle in angles:

        result.append(
            calculate_hand_force(
                angle,
                cg,
                mass,
                springs,
                hand_point
            )
        )

    return np.array(result)


def choose_physical_candidate(
    candidates,
    lid_length,
    lid_thickness
):
    """
    Z více matematických řešení vybere řešení,
    které je nejblíže ploše víka.
    """

    if not candidates:
        return None

    def score(p):

        x_penalty = 0

        if p[0] < 0:
            x_penalty += abs(p[0])

        if p[0] > lid_length:
            x_penalty += abs(p[0] - lid_length)

        y_penalty = abs(p[1])

        return x_penalty * 5 + y_penalty

    return min(
        candidates,
        key=score
    )


# ============================================================
# SIDEBAR – VSTUPY
# ============================================================

st.sidebar.header("Geometrie víka")

lid_length = st.sidebar.number_input(
    "Délka víka [mm]",
    min_value=100.0,
    value=1000.0,
    step=10.0
)

lid_thickness = st.sidebar.number_input(
    "Výška / tloušťka víka [mm]",
    min_value=1.0,
    value=50.0,
    step=5.0
)

mass = st.sidebar.number_input(
    "Hmotnost víka [kg]",
    min_value=0.1,
    value=20.0,
    step=0.5
)

st.sidebar.subheader("Těžiště CG")

cg_x = st.sidebar.number_input(
    "CG X [mm]",
    value=500.0,
    step=10.0
)

cg_y = st.sidebar.number_input(
    "CG Y [mm]",
    value=0.0,
    step=10.0
)

cg = np.array([
    cg_x,
    cg_y
])


max_angle_deg = st.sidebar.slider(
    "Maximální úhel otevření [°]",
    min_value=10,
    max_value=170,
    value=90
)

max_angle = math.radians(
    max_angle_deg
)


st.sidebar.header("Konfigurace")

configuration = st.sidebar.radio(
    "Konfigurace vzpěr",
    [
        "2 hlavní vzpěry",
        "2 hlavní + 2 pomocné"
    ]
)


# ============================================================
# HLAVNÍ VZPĚRA
# ============================================================

st.sidebar.header("Hlavní vzpěra – 1 ks")

main_base_x = st.sidebar.number_input(
    "Vana X [mm]",
    value=250.0,
    step=10.0,
    key="main_x"
)

main_base_y = st.sidebar.number_input(
    "Vana Y [mm]",
    value=-250.0,
    step=10.0,
    key="main_y"
)

main_closed_length = st.sidebar.number_input(
    "Zasunutá délka [mm]",
    min_value=20.0,
    value=400.0,
    step=5.0,
    key="main_length"
)

main_stroke = st.sidebar.number_input(
    "Zdvih [mm]",
    min_value=1.0,
    value=150.0,
    step=5.0,
    key="main_stroke"
)


main_base = np.array([
    main_base_x,
    main_base_y
])


# ============================================================
# POMOCNÁ VZPĚRA
# ============================================================

aux_base = None
aux_closed_length = None
aux_stroke = None

if configuration == "2 hlavní + 2 pomocné":

    st.sidebar.header("Pomocná vzpěra – 1 ks")

    aux_base_x = st.sidebar.number_input(
        "Vana X [mm]",
        value=100.0,
        step=10.0,
        key="aux_x"
    )

    aux_base_y = st.sidebar.number_input(
        "Vana Y [mm]",
        value=-150.0,
        step=10.0,
        key="aux_y"
    )

    aux_closed_length = st.sidebar.number_input(
        "Zasunutá délka [mm]",
        min_value=20.0,
        value=300.0,
        step=5.0,
        key="aux_length"
    )

    aux_stroke = st.sidebar.number_input(
        "Zdvih [mm]",
        min_value=1.0,
        value=100.0,
        step=5.0,
        key="aux_stroke"
    )

    aux_base = np.array([
        aux_base_x,
        aux_base_y
    ])


# ============================================================
# RUKA
# ============================================================

st.sidebar.header("Manipulace")

hand_distance = st.sidebar.number_input(
    "Vzdálenost ruky od pantu [mm]",
    min_value=50.0,
    value=float(lid_length),
    step=10.0
)

hand_point = np.array([
    hand_distance,
    0.0
])


target_hand_kg = st.sidebar.number_input(
    "Požadovaná síla při 0° [kg]",
    min_value=0.1,
    value=5.0,
    step=0.5
)

target_hand_force = (
    target_hand_kg * 9.81
)


# ============================================================
# VÝPOČET HLAVNÍHO ČEPU
# ============================================================

main_candidates = solve_main_pin(
    main_base,
    main_closed_length,
    main_stroke,
    max_angle
)

main_pin = choose_physical_candidate(
    main_candidates,
    lid_length,
    lid_thickness
)


# ============================================================
# MRTVÝ BOD
# ============================================================

dead_angle = None

if abs(cg_x) > 1e-9 or abs(cg_y) > 1e-9:

    # x světové souřadnice CG:
    #
    # x = cx*cos(theta) - cy*sin(theta)
    #
    # x = 0
    #
    # tan(theta) = cx / cy

    dead_angle = math.atan2(
        cg_x,
        cg_y
    )

    if dead_angle < 0:
        dead_angle += math.pi

    if dead_angle > max_angle:
        dead_angle = None


# ============================================================
# VÝPOČET POMOCNÉHO ČEPU
# ============================================================

aux_pin = None

if (
    configuration == "2 hlavní + 2 pomocné"
    and dead_angle is not None
):

    aux_candidates = solve_aux_pin(
        aux_base,
        aux_closed_length,
        dead_angle
    )

    aux_pin = choose_physical_candidate(
        aux_candidates,
        lid_length,
        lid_thickness
    )


# ============================================================
# KONTROLA GEOMETRIE
# ============================================================

st.header("Výsledky geometrie")


if main_pin is None:

    st.error(
        "Hlavní vzpěru nelze pro zadanou geometrii vyřešit. "
        "Změň polohu vany, délku vzpěry, zdvih nebo maximální úhel."
    )

else:

    main_length_max = distance(
        main_base,
        rotate_point(
            main_pin,
            max_angle
        )
    )

    main_real_stroke = (
        main_length_max
        - main_closed_length
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Hlavní čep X",
        f"{main_pin[0]:.1f} mm"
    )

    c2.metric(
        "Hlavní čep Y",
        f"{main_pin[1]:.1f} mm"
    )

    c3.metric(
        "Délka při max. otevření",
        f"{main_length_max:.1f} mm"
    )

    c4.metric(
        "Skutečný zdvih",
        f"{main_real_stroke:.1f} mm"
    )


if configuration == "2 hlavní + 2 pomocné":

    if dead_angle is None:

        st.warning(
            "Mrtvý bod CG leží mimo rozsah otevření."
        )

    elif aux_pin is None:

        st.error(
            "Pomocnou vzpěru nelze geometricky vyřešit."
        )

    else:

        aux_length_max = distance(
            aux_base,
            rotate_point(
                aux_pin,
                max_angle
            )
        )

        aux_real_stroke = (
            aux_length_max
            - aux_closed_length
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Pomocný čep X",
            f"{aux_pin[0]:.1f} mm"
        )

        c2.metric(
            "Pomocný čep Y",
            f"{aux_pin[1]:.1f} mm"
        )

        c3.metric(
            "Mrtvý bod",
            f"{math.degrees(dead_angle):.1f}°"
        )

        c4.metric(
            "Pomocný skutečný zdvih",
            f"{aux_real_stroke:.1f} mm"
        )


# ============================================================
# VÝPOČET POTŘEBNÝCH SIL
# ============================================================

st.header("Potřebná síla vzpěr")


main_unit_torque = 0.0

if main_pin is not None:

    main_unit_torque = spring_torque(
        main_base,
        main_pin,
        1.0
    )


aux_unit_torque = 0.0

if aux_pin is not None:

    aux_unit_torque = spring_torque(
        aux_base,
        aux_pin,
        1.0
    )


gravity_moment_0 = gravity_torque(
    cg,
    0.0,
    mass
)


hand_moment_0 = hand_distance * target_hand_force


required_spring_moment = (
    -(gravity_moment_0 + hand_moment_0)
)


# ============================================================
# ROZDĚLENÍ SIL
# ============================================================

aux_share = 0.0

if (
    configuration == "2 hlavní + 2 pomocné"
    and aux_pin is not None
):

    aux_share = st.slider(
        "Podíl momentu nesený pomocnými vzpěrami",
        min_value=0.0,
        max_value=0.8,
        value=0.25,
        step=0.05
    )


main_total_moment = (
    required_spring_moment
    * (1 - aux_share)
)


aux_total_moment = (
    required_spring_moment
    * aux_share
)


if abs(main_unit_torque) > 1e-9:

    main_force = (
        main_total_moment
        / (2 * main_unit_torque)
    )

else:

    main_force = np.nan


if (
    aux_pin is not None
    and abs(aux_unit_torque) > 1e-9
):

    aux_force = (
        aux_total_moment
        / (2 * aux_unit_torque)
    )

else:

    aux_force = 0.0


# ============================================================
# METRIKY
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Hlavní vzpěra – 1 ks",
    f"{main_force:.0f} N"
)

c2.metric(
    "Pomocná vzpěra – 1 ks",
    f"{aux_force:.0f} N"
)

c3.metric(
    "Cílová síla do ruky",
    f"{target_hand_force:.0f} N"
)

c4.metric(
    "Cílová síla do ruky",
    f"{target_hand_kg:.1f} kg"
)


if main_force < 0:

    st.warning(
        "Výsledná síla hlavní vzpěry je záporná. "
        "Geometrie vytváří moment v opačném směru. "
        "Zkontroluj umístění čepů."
    )


if aux_force < 0:

    st.warning(
        "Výsledná síla pomocné vzpěry je záporná."
    )


# ============================================================
# SESTAVENÍ VZPĚR
# ============================================================

springs = []

if (
    main_pin is not None
    and np.isfinite(main_force)
    and main_force > 0
):

    springs.append({
        "name": "Hlavní",
        "base": main_base,
        "pin": main_pin,
        "force": main_force
    })


if (
    aux_pin is not None
    and np.isfinite(aux_force)
    and aux_force > 0
):

    springs.append({
        "name": "Pomocná",
        "base": aux_base,
        "pin": aux_pin,
        "force": aux_force
    })


# ============================================================
# INTERAKTIVNÍ ÚHEL
# ============================================================

st.header("Vizualizace")

view_angle_deg = st.slider(
    "Aktuální úhel víka",
    min_value=0,
    max_value=max_angle_deg,
    value=0,
    step=1
)

view_angle = math.radians(
    view_angle_deg
)


# ============================================================
# GRAF GEOMETRIE
# ============================================================

fig_geometry, ax_geometry = plt.subplots(
    figsize=(8, 6)
)


# Víko
lid_points = np.array([
    [0, -lid_thickness / 2],
    [lid_length, -lid_thickness / 2],
    [lid_length, lid_thickness / 2],
    [0, lid_thickness / 2],
    [0, -lid_thickness / 2]
])


lid_world = np.array([
    rotate_point(
        p,
        view_angle
    )
    for p in lid_points
])


ax_geometry.plot(
    lid_world[:, 0],
    lid_world[:, 1],
    linewidth=2,
    label="Víko"
)


# Pant
ax_geometry.scatter(
    0,
    0,
    s=80,
    marker="x",
    label="Pant",
    zorder=10
)


# CG
cg_world = rotate_point(
    cg,
    view_angle
)


ax_geometry.scatter(
    cg_world[0],
    cg_world[1],
    s=100,
    marker="o",
    label="CG",
    zorder=10
)


ax_geometry.text(
    cg_world[0],
    cg_world[1],
    " CG",
    fontsize=10
)


# ============================================================
# HLAVNÍ VZPĚRY
# ============================================================

if main_pin is not None:

    main_pin_world = rotate_point(
        main_pin,
        view_angle
    )

    ax_geometry.plot(
        [
            main_base[0],
            main_pin_world[0]
        ],
        [
            main_base[1],
            main_pin_world[1]
        ],
        linewidth=4,
        label="Hlavní vzpěra"
    )

    ax_geometry.scatter(
        main_base[0],
        main_base[1],
        s=50
    )

    ax_geometry.scatter(
        main_pin_world[0],
        main_pin_world[1],
        s=50
    )


# ============================================================
# POMOCNÉ VZPĚRY
# ============================================================

if aux_pin is not None:

    aux_pin_world = rotate_point(
        aux_pin,
        view_angle
    )

    ax_geometry.plot(
        [
            aux_base[0],
            aux_pin_world[0]
        ],
        [
            aux_base[1],
            aux_pin_world[1]
        ],
        linewidth=4,
        linestyle="--",
        label="Pomocná vzpěra"
    )

    ax_geometry.scatter(
        aux_base[0],
        aux_base[1],
        s=50
    )

    ax_geometry.scatter(
        aux_pin_world[0],
        aux_pin_world[1],
        s=50
    )


# ============================================================
# MRTVÝ BOD
# ============================================================

if dead_angle is not None:

    dead_deg = math.degrees(
        dead_angle
    )

    if dead_deg <= max_angle_deg:

        cg_dead = rotate_point(
            cg,
            dead_angle
        )

        ax_geometry.scatter(
            cg_dead[0],
            cg_dead[1],
            s=80,
            marker="D",
            label=f"Mrtvý bod {dead_deg:.1f}°"
        )


ax_geometry.set_aspect(
    "equal",
    adjustable="datalim"
)

ax_geometry.grid(
    True,
    alpha=0.25
)

ax_geometry.set_xlabel(
    "X [mm]"
)

ax_geometry.set_ylabel(
    "Y [mm]"
)

ax_geometry.set_title(
    f"Geometrie při {view_angle_deg}°"
)

ax_geometry.legend(
    loc="best"
)


# ============================================================
# PROFIL SÍLY
# ============================================================

angles_deg = np.linspace(
    0,
    max_angle_deg,
    300
)


profile_N = calculate_profile(
    np.radians(angles_deg),
    cg,
    mass,
    springs,
    hand_point
)


fig_force, ax_force = plt.subplots(
    figsize=(8, 6)
)


positive = np.where(
    profile_N >= 0,
    profile_N,
    np.nan
)

negative = np.where(
    profile_N < 0,
    profile_N,
    np.nan
)


ax_force.plot(
    angles_deg,
    positive,
    linewidth=2,
    label="Zvedání víka"
)


ax_force.plot(
    angles_deg,
    negative,
    linewidth=2,
    label="Brždění / držení"
)


ax_force.axhline(
    0,
    linewidth=1
)


ax_force.axvline(
    view_angle_deg,
    linestyle=":",
    linewidth=1
)


if dead_angle is not None:

    dead_deg = math.degrees(
        dead_angle
    )

    if dead_deg <= max_angle_deg:

        ax_force.axvline(
            dead_deg,
            linestyle="--",
            linewidth=1,
            label=f"Mrtvý bod {dead_deg:.1f}°"
        )


ax_force.set_xlabel(
    "Úhel otevření [°]"
)

ax_force.set_ylabel(
    "Síla do ruky [N]"
)

ax_force.set_title(
    "Profil síly do ruky"
)

ax_force.grid(
    True,
    alpha=0.25
)

ax_force.legend(
    loc="best"
)


# ============================================================
# GRAFY VEDLE SEBE
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.pyplot(
        fig_geometry,
        clear_figure=True
    )


with col2:

    st.pyplot(
        fig_force,
        clear_figure=True
    )


# ============================================================
# DETAILNÍ KONTROLA
# ============================================================

st.header("Kontrola v jednotlivých úhlech")


check_angles = sorted(
    set([
        0,
        min(30, max_angle_deg),
        min(45, max_angle_deg),
        min(50, max_angle_deg),
        min(60, max_angle_deg),
        max_angle_deg,
        view_angle_deg
    ])
)


rows = []


for angle_deg in check_angles:

    angle = math.radians(
        angle_deg
    )

    main_length = np.nan
    aux_length = np.nan

    if main_pin is not None:

        main_length = distance(
            main_base,
            rotate_point(
                main_pin,
                angle
            )
        )

    if aux_pin is not None:

        aux_length = distance(
            aux_base,
            rotate_point(
                aux_pin,
                angle
            )
        )

    hand_force = calculate_hand_force(
        angle,
        cg,
        mass,
        springs,
        hand_point
    )

    rows.append({
        "Úhel [°]": angle_deg,

        "Hlavní délka [mm]":
            round(main_length, 1),

        "Hlavní zdvih [mm]":
            round(
                main_length - main_closed_length,
                1
            ) if np.isfinite(main_length)
            else np.nan,

        "Pomocná délka [mm]":
            round(aux_length, 1)
            if np.isfinite(aux_length)
            else np.nan,

        "Pomocný zdvih [mm]":
            round(
                aux_length - aux_closed_length,
                1
            )
            if np.isfinite(aux_length)
            else np.nan,

        "Síla do ruky [N]":
            round(hand_force, 1),

        "Síla do ruky [kg]":
            round(
                hand_force / 9.81,
                2
            )
    })


st.dataframe(
    rows,
    use_container_width=True
)


# ============================================================
# DALŠÍ KONTROLY
# ============================================================

st.header("Kontrola pracovního rozsahu")


if main_pin is not None:

    main_lengths = np.array([
        distance(
            main_base,
            rotate_point(
                main_pin,
                math.radians(a)
            )
        )
        for a in angles_deg
    ])

    main_strokes = (
        main_lengths
        - main_closed_length
    )

    if np.min(main_strokes) < -0.1:

        st.error(
            "Hlavní vzpěra se v části rozsahu dostává "
            "pod svou zadanou zasunutou délku."
        )

    if np.max(main_strokes) > main_stroke + 0.1:

        st.warning(
            "Hlavní vzpěra překračuje zadaný zdvih."
        )


if aux_pin is not None:

    aux_lengths = np.array([
        distance(
            aux_base,
            rotate_point(
                aux_pin,
                math.radians(a)
            )
        )
        for a in angles_deg
    ])

    aux_strokes = (
        aux_lengths
        - aux_closed_length
    )

    if np.min(aux_strokes) < -0.1:

        st.error(
            "Pomocná vzpěra se dostává pod svou "
            "zadanou zasunutou délku."
        )

    if np.max(aux_strokes) > aux_stroke + 0.1:

        st.warning(
            "Pomocná vzpěra překračuje zadaný zdvih."
        )


# ============================================================
# POZNÁMKY
# ============================================================

st.header("Poznámky k výpočtu")

st.markdown(
    """
**Hlavní vzpěra**

Čep na víku je dopočítán přesně z podmínek:

- délka vzpěry při 0° = zadaná zasunutá délka,
- délka vzpěry při maximálním otevření =
  zasunutá délka + zdvih.

**Pomocná vzpěra**

Čep je dopočítán z podmínky, že v okamžiku,
kdy CG přejde přes osu pantu, prochází osa pomocné
vzpěry přesně bodem `[0, 0]`.

**Síly**

Síla je počítána z momentové rovnováhy kolem pantu.
Výsledná síla je síla **jednoho kusu** vzpěry.

U konfigurace 2 + 2 se tedy moment každého typu
násobí dvěma.

**Pozor pro reálný návrh**

Tento výpočet zatím neuvažuje:

- změnu síly vzpěry během zdvihu,
- tlak plynu,
- tření pístnice,
- teplotní závislost,
- výrobní tolerance,
- dynamické rázy,
- bezpečnostní koeficient,
- skutečnou polohu a směr síly ruky,
- případnou nelinearitu konstrukce.

Pro katalogový výběr je proto vhodné následně pracovat
s hodnotami **F1 / F2** konkrétní plynové vzpěry.
"""
)
