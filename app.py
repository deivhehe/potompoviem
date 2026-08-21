import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# NÁVRH PLYNOVÝCH VZPĚR PRO VÝKLOPNÉ VÍKO
#
# Souřadný systém:
#   pant = [0, 0]
#   zavřené víko směřuje doleva a nahoru
#   0° = zavřeno
#   kladný úhel = otevírání víka po směru hodinových ručiček
#
# Tím odpovídá nákres přibližně konstrukci typu SolidWorks:
#
#                  levá strana víka
#        ┌───────────────────────────●
#        │                           čep
#        │
#        │
#        └───────────────────────────X  pant [0,0]
#
# při 90° je víko přibližně svisle nahoru.
# ============================================================


st.set_page_config(
    page_title="Návrh plynových vzpěr",
    layout="wide",
)


# ============================================================
# MATEMATIKA
# ============================================================

def rotate_clockwise(point, angle):
    """
    Rotace bodu kolem pantu ve směru hodinových ručiček.
    """

    c = math.cos(angle)
    s = math.sin(angle)

    x = point[0]
    y = point[1]

    return np.array([
        c * x + s * y,
        -s * x + c * y
    ])


def rotate_counterclockwise(point, angle):
    """
    Inverzní rotace.
    """

    return rotate_clockwise(point, -angle)


def distance(a, b):
    return float(np.linalg.norm(a - b))


def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def spring_torque(base, lid_pin, force):
    """
    Moment plynové vzpěry kolem pantu.

    Vzpěra působí tlakem z bodu na víku směrem
    k bodu uchycení na vaně.
    """

    vector = base - lid_pin
    length = np.linalg.norm(vector)

    if length < 1e-9:
        return 0.0

    force_vector = force * vector / length

    return cross_2d(lid_pin, force_vector)


def gravity_torque(cg, angle, mass):
    """
    Moment gravitace kolem pantu.
    """

    cg_world = rotate_clockwise(
        cg,
        angle
    )

    gravity = np.array([
        0.0,
        -mass * 9.81
    ])

    return cross_2d(
        cg_world,
        gravity
    )


def circle_intersections(c1, r1, c2, r2):
    """
    Průsečíky dvou kružnic.
    """

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]

    d = math.hypot(dx, dy)

    if d < 1e-9:
        return []

    if d > r1 + r2 + 1e-9:
        return []

    if d < abs(r1 - r2) - 1e-9:
        return []

    a = (
        r1**2
        - r2**2
        + d**2
    ) / (2 * d)

    h2 = r1**2 - a**2

    if h2 < -1e-9:
        return []

    h = math.sqrt(
        max(0.0, h2)
    )

    xm = c1[0] + a * dx / d
    ym = c1[1] + a * dy / d

    rx = -dy * h / d
    ry = dx * h / d

    return [
        np.array([
            xm + rx,
            ym + ry
        ]),
        np.array([
            xm - rx,
            ym - ry
        ])
    ]


def point_on_lid(point, length, height, tolerance=1.0):
    """
    Kontrola, jestli bod leží uvnitř obdélníku víka.

    Víko:
        X = -length ... 0
        Y = 0 ... height

    Pant je v pravém dolním rohu [0,0].
    """

    return (
        -tolerance <= point[0] <= tolerance
        and
        -tolerance <= point[1] <= height + tolerance
    ) or (
        -length - tolerance <= point[0] <= 0 + tolerance
        and
        0 - tolerance <= point[1] <= height + tolerance
    )


def pin_inside_lid(point, length, height):
    """
    Přísnější kontrola bodu na ploše víka.
    """

    return (
        -length <= point[0] <= 0
        and
        0 <= point[1] <= height
    )


# ============================================================
# HLAVNÍ VZPĚRA
# ============================================================

def solve_main_pin(
    base,
    closed_length,
    stroke,
    max_angle,
    lid_length,
    lid_height
):
    """
    Najde čep hlavní vzpěry na víku.

    Podmínky:

    při 0°:
        délka = closed_length

    při max. úhlu:
        délka = closed_length + stroke

    Navíc musí být čep fyzicky na víku.
    """

    open_length = (
        closed_length + stroke
    )

    # Při otevření:
    #
    # |R(P) - A| = open_length
    #
    # po převedení zpět do soustavy zavřeného víka:
    #
    # |P - R^-1(A)| = open_length

    base_back = rotate_counterclockwise(
        base,
        max_angle
    )

    candidates = circle_intersections(
        base,
        closed_length,
        base_back,
        open_length
    )

    physical = [
        p
        for p in candidates
        if pin_inside_lid(
            p,
            lid_length,
            lid_height
        )
    ]

    return physical


# ============================================================
# MRTVÝ BOD CG
# ============================================================

def calculate_dead_angle(cg):
    """
    Najde úhel, při kterém CG leží na svislé ose pantu.

    Ve světových souřadnicích:
        X_CG = 0

    Pro clockwise rotaci:

        X = X0*cos(theta) + Y0*sin(theta)

    tedy:

        tan(theta) = -X0 / Y0
    """

    x = cg[0]
    y = cg[1]

    if abs(x) < 1e-9:
        return 0.0

    if abs(y) < 1e-9:

        if x < 0:
            return math.pi / 2

        return math.pi / 2

    angle = math.atan2(
        -x,
        y
    )

    # Hledáme první kladné řešení.
    if angle < 0:
        angle += math.pi

    return angle


# ============================================================
# POMOCNÁ VZPĚRA
# ============================================================

def solve_aux_pin(
    base,
    closed_length,
    dead_angle,
    lid_length,
    lid_height
):
    """
    Pomocná vzpěra:

    1. Při 0° je plně zasunutá:
           L = closed_length

    2. V mrtvém bodě CG musí osa vzpěry
       procházet přesně přes pant [0,0].

    To znamená, že při mrtvém bodě musí být
    horní čep na přímce:

           base -------- pant

    a zároveň musí být od základny vzdálený
    přesně closed_length.

    Z tohoto bodu se dopočítá jeho poloha
    na víku při 0°.

    Pomocná vzpěra NEMÁ povinně daný zdvih.
    Její délka při otevření je výsledkem geometrie.
    """

    base_to_hinge = -base

    d = np.linalg.norm(
        base_to_hinge
    )

    if d < 1e-9:
        return []

    direction = (
        base_to_hinge / d
    )

    # Bod na přímce base -> pant
    q1 = (
        base
        + direction * closed_length
    )

    q2 = (
        base
        - direction * closed_length
    )

    candidates = []

    for q in [q1, q2]:

        # Q je poloha čepu v mrtvém bodě.
        #
        # Potřebujeme jeho polohu v zavřeném
        # souřadném systému.

        pin0 = rotate_counterclockwise(
            q,
            dead_angle
        )

        if pin_inside_lid(
            pin0,
            lid_length,
            lid_height
        ):

            candidates.append(
                pin0
            )

    return candidates


# ============================================================
# VÝPOČET SÍLY DO RUKY
# ============================================================

def calculate_hand_force(
    angle,
    cg,
    mass,
    springs,
    hand_point
):

    hand_world = rotate_clockwise(
        hand_point,
        angle
    )

    hand_radius = np.linalg.norm(
        hand_world
    )

    if hand_radius < 1e-9:
        return np.nan

    total_moment = gravity_torque(
        cg,
        angle,
        mass
    )

    for spring in springs:

        pin_world = rotate_clockwise(
            spring["pin"],
            angle
        )

        total_moment += spring_torque(
            spring["base"],
            pin_world,
            spring["force"]
        )

    # Síla ruky je tečná.
    #
    # Kladná = uživatel musí zvedat.
    # Záporná = víko samo pomáhá otevírat /
    # uživatel musí brzdit zavírání.

    return -total_moment / hand_radius


# ============================================================
# VSTUPY
# ============================================================

st.sidebar.header("Geometrie víka")

lid_length = st.sidebar.number_input(
    "Délka víka [mm]",
    min_value=100.0,
    value=1100.0,
    step=10.0
)

lid_height = st.sidebar.number_input(
    "Výška víka [mm]",
    min_value=10.0,
    value=400.0,
    step=10.0
)

mass = st.sidebar.number_input(
    "Hmotnost víka [kg]",
    min_value=0.1,
    value=20.0,
    step=0.5
)


st.sidebar.subheader(
    "Těžiště CG vůči pantu"
)

cg_x = st.sidebar.number_input(
    "CG X [mm]",
    value=-550.0,
    step=10.0,
    help="Záporná hodnota = CG je směrem doleva od pantu."
)

cg_y = st.sidebar.number_input(
    "CG Y [mm]",
    value=200.0,
    step=10.0,
    help="Kladná hodnota = CG je nad spodní hranou víka."
)

cg = np.array([
    cg_x,
    cg_y
])


max_angle_deg = st.slider(
    "Maximální úhel otevření [°]",
    min_value=10,
    max_value=150,
    value=90
)

max_angle = math.radians(
    max_angle_deg
)


# ============================================================
# KONFIGURACE
# ============================================================

st.sidebar.header("Konfigurace")

configuration = st.sidebar.radio(
    "Vzpěry",
    [
        "2 hlavní vzpěry",
        "2 hlavní + 2 pomocné"
    ]
)


# ============================================================
# HLAVNÍ VZPĚRA
# ============================================================

st.sidebar.header(
    "Hlavní vzpěra – 1 ks"
)

main_base_x = st.sidebar.number_input(
    "Spodní čep na vaně X [mm]",
    value=-150.0,
    step=10.0,
    key="main_base_x"
)

main_base_y = st.sidebar.number_input(
    "Spodní čep na vaně Y [mm]",
    value=-400.0,
    step=10.0,
    key="main_base_y"
)

main_closed_length = st.sidebar.number_input(
    "Celková délka při 0° [mm]",
    min_value=20.0,
    value=450.0,
    step=5.0,
    help="Délka celé vzpěry v zavřeném stavu."
)

main_stroke = st.sidebar.number_input(
    "Zdvih [mm]",
    min_value=1.0,
    value=180.0,
    step=5.0,
    help="Při maximálním otevření bude délka = délka při 0° + zdvih."
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

if configuration == "2 hlavní + 2 pomocné":

    st.sidebar.header(
        "Pomocná vzpěra – 1 ks"
    )

    aux_base_x = st.sidebar.number_input(
        "Spodní čep na vaně X [mm]",
        value=-950.0,
        step=10.0,
        key="aux_base_x"
    )

    aux_base_y = st.sidebar.number_input(
        "Spodní čep na vaně Y [mm]",
        value=200.0,
        step=10.0,
        key="aux_base_y"
    )

    aux_closed_length = st.sidebar.number_input(
        "Celková délka při 0° [mm]",
        min_value=20.0,
        value=320.0,
        step=5.0,
        help=(
            "Pomocná vzpěra je v zavřeném stavu "
            "plně zasunutá. Její délka při otevření "
            "se dopočítá z geometrie."
        )
    )

    aux_base = np.array([
        aux_base_x,
        aux_base_y
    ])


# ============================================================
# SÍLA DO RUKY
# ============================================================

st.sidebar.header(
    "Síla do ruky"
)

target_hand_kg = st.sidebar.number_input(
    "Cílová síla při zavřeném víku [kg]",
    min_value=0.1,
    value=5.0,
    step=0.5
)

target_hand_force = (
    target_hand_kg * 9.81
)


hand_distance = st.sidebar.number_input(
    "Vzdálenost působení ruky od pantu [mm]",
    min_value=50.0,
    value=float(lid_length),
    step=10.0
)

hand_point = np.array([
    -hand_distance,
    0.0
])


# ============================================================
# VÝPOČET HLAVNÍHO ČEPU
# ============================================================

main_candidates = solve_main_pin(
    main_base,
    main_closed_length,
    main_stroke,
    max_angle,
    lid_length,
    lid_height
)


# ============================================================
# VÝPOČET MRTVÉHO BODU
# ============================================================

dead_angle = calculate_dead_angle(
    cg
)


dead_angle_deg = math.degrees(
    dead_angle
)


# ============================================================
# VÝPOČET POMOCNÉHO ČEPU
# ============================================================

aux_candidates = []

aux_pin = None

if (
    configuration == "2 hlavní + 2 pomocné"
    and aux_base is not None
):

    aux_candidates = solve_aux_pin(
        aux_base,
        aux_closed_length,
        dead_angle,
        lid_length,
        lid_height
    )

    if aux_candidates:

        # Vybereme řešení nejblíže středu víka.
        aux_pin = min(
            aux_candidates,
            key=lambda p: abs(
                p[1] - lid_height / 2
            )
        )


# ============================================================
# VÝBĚR HLAVNÍHO ČEPU
# ============================================================

main_pin = None

if main_candidates:

    # Preferujeme bod více uvnitř víka,
    # ne úplně na hraně.

    main_pin = min(
        main_candidates,
        key=lambda p:
            abs(p[1] - lid_height / 2)
    )


# ============================================================
# VÝSLEDKY GEOMETRIE
# ============================================================

st.header(
    "Výsledky – pozice horních čepů"
)


if main_pin is None:

    st.error(
        "❌ Hlavní čep nelze umístit na víko "
        "pro zadanou délku, zdvih, základní čep "
        "a maximální úhel."
    )

else:

    main_open_pin = rotate_clockwise(
        main_pin,
        max_angle
    )

    main_open_length = distance(
        main_base,
        main_open_pin
    )

    main_real_stroke = (
        main_open_length
        - main_closed_length
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Hlavní čep X",
            f"{main_pin[0]:.1f} mm"
        )

    with c2:
        st.metric(
            "Hlavní čep Y",
            f"{main_pin[1]:.1f} mm"
        )

    with c3:
        st.metric(
            "Délka při max. úhlu",
            f"{main_open_length:.1f} mm"
        )

    with c4:
        st.metric(
            "Skutečný zdvih",
            f"{main_real_stroke:.1f} mm"
        )

    if abs(
        main_real_stroke - main_stroke
    ) > 0.5:

        st.warning(
            "Geometrie nedokázala splnit současně "
            "požadovanou zasunutou délku a zdvih."
        )


# ============================================================
# POMOCNÁ VZPĚRA – VÝSLEDKY
# ============================================================

if configuration == "2 hlavní + 2 pomocné":

    if dead_angle_deg > max_angle_deg:

        st.warning(
            f"Mrtvý bod CG je až při "
            f"{dead_angle_deg:.1f}°, "
            f"což je mimo maximální otevření "
            f"{max_angle_deg}°."
        )

    if aux_pin is None:

        st.error(
            "❌ Pomocný čep nelze umístit na víko "
            "tak, aby při mrtvém bodě osa vzpěry "
            "procházela přesně pantem."
        )

    else:

        aux_dead_pin = rotate_clockwise(
            aux_pin,
            dead_angle
        )

        aux_dead_length = distance(
            aux_base,
            aux_dead_pin
        )

        aux_open_pin = rotate_clockwise(
            aux_pin,
            max_angle
        )

        aux_open_length = distance(
            aux_base,
            aux_open_pin
        )

        aux_open_stroke = (
            aux_open_length
            - aux_closed_length
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Pomocný čep X",
                f"{aux_pin[0]:.1f} mm"
            )

        with c2:
            st.metric(
                "Pomocný čep Y",
                f"{aux_pin[1]:.1f} mm"
            )

        with c3:
            st.metric(
                "Mrtvý bod CG",
                f"{dead_angle_deg:.1f}°"
            )

        with c4:
            st.metric(
                "Pomocná délka při max.",
                f"{aux_open_length:.1f} mm"
            )

        st.info(
            f"Pomocná vzpěra je při 0° dlouhá "
            f"{aux_closed_length:.1f} mm. "
            f"V mrtvém bodě ({dead_angle_deg:.1f}°) "
            f"má délku {aux_dead_length:.1f} mm. "
            f"Při maximálním otevření má "
            f"{aux_open_length:.1f} mm "
            f"(změna délky {aux_open_stroke:+.1f} mm)."
        )


# ============================================================
# KONTROLA, KDE JSOU ČEPY
# ============================================================

if main_pin is not None:

    if not pin_inside_lid(
        main_pin,
        lid_length,
        lid_height
    ):

        st.error(
            "Hlavní čep není uvnitř plochy víka."
        )


if aux_pin is not None:

    if not pin_inside_lid(
        aux_pin,
        lid_length,
        lid_height
    ):

        st.error(
            "Pomocný čep není uvnitř plochy víka."
        )


# ============================================================
# VÝPOČET SIL
# ============================================================

st.header(
    "Potřebné katalogové síly"
)


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
    0,
    mass
)


# Síla ruky při zavřeném víku.
#
# Ruka působí ve směru otevření.
#
# Bod ruky je vlevo od pantu.
#
# Pro kladnou otevírací sílu:
# moment je záporný / podle našeho znaménka.
hand_moment_0 = (
    hand_distance
    * target_hand_force
)


required_spring_moment = -(
    gravity_moment_0
    + hand_moment_0
)


# ============================================================
# ROZDĚLENÍ MOMENTU
# ============================================================

aux_share = 0.0

if (
    configuration == "2 hlavní + 2 pomocné"
    and aux_pin is not None
):

    aux_share = st.slider(
        "Podíl celkového momentu nesený pomocnými vzpěrami",
        min_value=0.0,
        max_value=0.8,
        value=0.25,
        step=0.05,
        help=(
            "0 % = veškerý moment nesou hlavní vzpěry. "
            "25 % = pomocné nesou přibližně čtvrtinu "
            "celkového momentu."
        )
    )


main_total_moment = (
    required_spring_moment
    * (1 - aux_share)
)


aux_total_moment = (
    required_spring_moment
    * aux_share
)


# 2 ks hlavních vzpěr
if abs(main_unit_torque) > 1e-9:

    main_force = (
        main_total_moment
        / (2 * main_unit_torque)
    )

else:

    main_force = np.nan


# 2 ks pomocných vzpěr
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
# METRIKY SIL
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    if np.isfinite(main_force):

        st.metric(
            "Hlavní vzpěra – 1 ks",
            f"{main_force:.0f} N"
        )

    else:

        st.metric(
            "Hlavní vzpěra – 1 ks",
            "NELZE"
        )


with c2:

    st.metric(
        "Pomocná vzpěra – 1 ks",
        (
            f"{aux_force:.0f} N"
            if aux_pin is not None
            else "—"
        )
    )


with c3:

    st.metric(
        "Síla do ruky při 0°",
        f"{target_hand_force:.0f} N"
    )


with c4:

    st.metric(
        "Mrtvý bod",
        f"{dead_angle_deg:.1f}°"
    )


# ============================================================
# VZPĚRY PRO SIMULACI
# ============================================================

springs = []


if (
    main_pin is not None
    and np.isfinite(main_force)
    and main_force > 0
):

    springs.append({
        "name": "Hlavní vzpěra",
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
        "name": "Pomocná vzpěra",
        "base": aux_base,
        "pin": aux_pin,
        "force": aux_force
    })


# ============================================================
# ÚHEL PRO ZOBRAZENÍ
# ============================================================

st.header(
    "Geometrie"
)


view_angle_deg = st.slider(
    "Zobrazený úhel víka",
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
    figsize=(9, 7)
)


# ------------------------------------------------------------
# VÍKO
# ------------------------------------------------------------

# Pant je v pravém dolním rohu.
#
# Zavřené víko:
#
#       (-L,H) ──────────── (0,H)
#          │                    │
#          │                    │
#       (-L,0) ──────────── (0,0) ← pant
#
# Víko je tedy vlevo od pantu.

lid_closed = np.array([
    [-lid_length, 0],
    [0, 0],
    [0, lid_height],
    [-lid_length, lid_height],
    [-lid_length, 0]
])


lid_world = np.array([
    rotate_clockwise(
        point,
        view_angle
    )
    for point in lid_closed
])


ax_geometry.plot(
    lid_world[:, 0],
    lid_world[:, 1],
    linewidth=2.5,
    label="Víko"
)


# ------------------------------------------------------------
# PANT
# ------------------------------------------------------------

ax_geometry.scatter(
    [0],
    [0],
    s=100,
    marker="x",
    linewidths=3,
    label="Pant [0,0]",
    zorder=20
)


# ------------------------------------------------------------
# CG
# ------------------------------------------------------------

cg_world = rotate_clockwise(
    cg,
    view_angle
)


ax_geometry.scatter(
    [cg_world[0]],
    [cg_world[1]],
    s=100,
    marker="o",
    label="CG",
    zorder=20
)


ax_geometry.text(
    cg_world[0] + 15,
    cg_world[1] + 15,
    "CG"
)


# ------------------------------------------------------------
# HLAVNÍ VZPĚRA
# ------------------------------------------------------------

if main_pin is not None:

    main_pin_world = rotate_clockwise(
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
        [main_base[0]],
        [main_base[1]],
        s=60,
        zorder=20
    )

    ax_geometry.scatter(
        [main_pin_world[0]],
        [main_pin_world[1]],
        s=60,
        zorder=20
    )


# ------------------------------------------------------------
# POMOCNÁ VZPĚRA
# ------------------------------------------------------------

if aux_pin is not None:

    aux_pin_world = rotate_clockwise(
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
        [aux_base[0]],
        [aux_base[1]],
        s=60,
        zorder=20
    )

    ax_geometry.scatter(
        [aux_pin_world[0]],
        [aux_pin_world[1]],
        s=60,
        zorder=20
    )


# ------------------------------------------------------------
# MRTVÝ BOD
# ------------------------------------------------------------

if dead_angle <= max_angle:

    cg_dead = rotate_clockwise(
        cg,
        dead_angle
    )

    ax_geometry.scatter(
        [cg_dead[0]],
        [cg_dead[1]],
        s=100,
        marker="D",
        label=(
            f"CG na ose pantu "
            f"({dead_angle_deg:.1f}°)"
        ),
        zorder=25
    )

    # Svislá osa pantu
    ax_geometry.axvline(
        0,
        linestyle=":",
        linewidth=1
    )


# ------------------------------------------------------------
# VZHLED
# ------------------------------------------------------------

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
# PROFIL DÉLEK VZPĚR
# ============================================================

angles_deg = np.linspace(
    0,
    max_angle_deg,
    300
)


main_lengths = []
aux_lengths = []


for angle_deg in angles_deg:

    angle = math.radians(
        angle_deg
    )

    if main_pin is not None:

        main_lengths.append(
            distance(
                main_base,
                rotate_clockwise(
                    main_pin,
                    angle
                )
            )
        )

    else:

        main_lengths.append(
            np.nan
        )


    if aux_pin is not None:

        aux_lengths.append(
            distance(
                aux_base,
                rotate_clockwise(
                    aux_pin,
                    angle
                )
            )
        )

    else:

        aux_lengths.append(
            np.nan
        )


main_lengths = np.array(
    main_lengths
)

aux_lengths = np.array(
    aux_lengths
)


# ============================================================
# PROFIL SÍLY
# ============================================================

profile_force = np.array([
    calculate_hand_force(
        math.radians(angle),
        cg,
        mass,
        springs,
        hand_point
    )
    for angle in angles_deg
])


fig_force, ax_force = plt.subplots(
    figsize=(9, 6)
)


positive = np.where(
    profile_force >= 0,
    profile_force,
    np.nan
)

negative = np.where(
    profile_force < 0,
    profile_force,
    np.nan
)


ax_force.plot(
    angles_deg,
    positive,
    linewidth=2,
    label="Zvedání – kladná síla"
)

ax_force.plot(
    angles_deg,
    negative,
    linewidth=2,
    label="Brždění / držení – záporná síla"
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


if dead_angle <= max_angle:

    ax_force.axvline(
        dead_angle_deg,
        linestyle="--",
        linewidth=1,
        label=f"Mrtvý bod ({dead_angle_deg:.1f}°)"
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
# KONTROLNÍ TABULKA
# ============================================================

st.header(
    "Kontrola délek a sil"
)


check_angles = sorted(
    set([
        0,
        min(20, max_angle_deg),
        min(40, max_angle_deg),
        min(45, max_angle_deg),
        min(50, max_angle_deg),
        min(60, max_angle_deg),
        min(80, max_angle_deg),
        max_angle_deg,
        view_angle_deg
    ])
)


rows = []


for angle_deg in check_angles:

    angle = math.radians(
        angle_deg
    )


    if main_pin is not None:

        main_length = distance(
            main_base,
            rotate_clockwise(
                main_pin,
                angle
            )
        )

    else:

        main_length = np.nan


    if aux_pin is not None:

        aux_length = distance(
            aux_base,
            rotate_clockwise(
                aux_pin,
                angle
            )
        )

    else:

        aux_length = np.nan


    hand_force = calculate_hand_force(
        angle,
        cg,
        mass,
        springs,
        hand_point
    )


    rows.append({
        "Úhel [°]":
            round(angle_deg, 1),

        "Délka hlavní [mm]":
            round(main_length, 1)
            if np.isfinite(main_length)
            else np.nan,

        "Zdvih hlavní [mm]":
            round(
                main_length
                - main_closed_length,
                1
            )
            if np.isfinite(main_length)
            else np.nan,

        "Délka pomocné [mm]":
            round(aux_length, 1)
            if np.isfinite(aux_length)
            else np.nan,

        "Zdvih pomocné [mm]":
            round(
                aux_length
                - aux_closed_length,
                1
            )
            if np.isfinite(aux_length)
            else np.nan,

        "Síla do ruky [N]":
            round(
                hand_force,
                1
            ),

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
# KONTROLY
# ============================================================

st.header(
    "Kontrola geometrie"
)


# ------------------------------------------------------------
# Hlavní vzpěra
# ------------------------------------------------------------

if main_pin is not None:

    main_error = abs(
        main_lengths[-1]
        - (
            main_closed_length
            + main_stroke
        )
    )

    if main_error < 0.5:

        st.success(
            "✓ Hlavní vzpěra splňuje podmínku: "
            "0° = zasunutá délka, "
            "max. úhel = zasunutá délka + zdvih."
        )

    else:

        st.error(
            "Hlavní vzpěra nesplňuje zadanou "
            "kombinaci délky a zdvihu."
        )


# ------------------------------------------------------------
# Pomocná vzpěra
# ------------------------------------------------------------

if aux_pin is not None:

    aux_dead_pin_check = rotate_clockwise(
        aux_pin,
        dead_angle
    )

    # Vektor od základny k čepu
    # musí být kolineární s vektorem
    # od základny k pantu.

    v1 = aux_dead_pin_check - aux_base
    v2 = -aux_base

    cross = abs(
        cross_2d(v1, v2)
    )

    if cross < 1.0:

        st.success(
            "✓ Pomocná vzpěra v mrtvém bodě "
            f"({dead_angle_deg:.1f}°) přesně prochází "
            "osou pantu."
        )

    else:

        st.error(
            "Pomocná vzpěra v mrtvém bodě "
            "neprochází osou pantu."
        )


# ------------------------------------------------------------
# Síla
# ------------------------------------------------------------

if np.isfinite(main_force):

    if main_force > 0:

        st.success(
            f"Hlavní vzpěra: {main_force:.0f} N / ks."
        )

    else:

        st.error(
            "Hlavní vzpěra vychází se zápornou silou. "
            "Je potřeba změnit geometrii."
        )


if aux_pin is not None:

    if aux_force > 0:

        st.success(
            f"Pomocná vzpěra: {aux_force:.0f} N / ks."
        )

    else:

        st.warning(
            "Pomocná vzpěra vychází se zápornou silou."
        )


# ============================================================
# INFORMAČNÍ PANEL
# ============================================================

st.header(
    "Logika výpočtu"
)

st.markdown(
    f"""
### Hlavní vzpěra

- při **0°** má přesně zadanou celkovou délku
  **{main_closed_length:.1f} mm**
- při **{max_angle_deg}°** má přesně
  **{main_closed_length + main_stroke:.1f} mm**
- horní čep je matematicky hledán pouze **uvnitř plochy víka**

### Pomocná vzpěra

- při **0°** má přesně zadanou celkovou délku
  **{aux_closed_length:.1f} mm**,
- její horní čep je hledán pouze **na víku**,
- v úhlu **{dead_angle_deg:.1f}°** musí osa vzpěry procházet
  přesně bodem **[0, 0] – pantem**,
- její délka při maximálním otevření se **nedává jako další
  pevná podmínka**, ale je výsledkem geometrie.

To je důležité: u pomocné vzpěry nelze obecně současně libovolně
nastavit zasunutou délku, zdvih, polohu spodního čepu a ještě
požadovat průchod osou pantu v přesně daném mrtvém bodě.
Bylo by to příliš mnoho geometrických podmínek.

### Znaménko síly do ruky

- **kladná hodnota** = musíš víko zvedat,
- **0 N** = mrtvý bod / rovnováha,
- **záporná hodnota** = vzpěry mají tendenci víko otevírat
  a při zavírání je musíš brzdit.
"""
)
