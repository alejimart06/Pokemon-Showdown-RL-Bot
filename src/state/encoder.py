"""
encoder.py
Convierte el estado de una batalla de poke-env en un vector numerico
que el agente RL puede procesar.

Estructura del vector de observacion:
  - Pokemon activo propio (HP, stats, tipos, estado, boosts, movimientos)
  - Analisis de combate (efectividad, daño estimado, KO flags)
  - Pokemon en reserva propios x5 (HP, tipos, disponible)
  - Pokemon activo enemigo (HP visible, tipos, estado, boosts)
  - Pokemon en reserva enemigos x5 (HP visible, tipos)
  - Condiciones del campo (clima, terreno, pantallas)
"""

import numpy as np
from poke_env.battle import AbstractBattle, Pokemon, Move, Weather, Field, SideCondition, Status, PokemonType

# Todos los tipos posibles (18)
TYPES = [
    PokemonType.NORMAL, PokemonType.FIRE, PokemonType.WATER, PokemonType.ELECTRIC,
    PokemonType.GRASS, PokemonType.ICE, PokemonType.FIGHTING, PokemonType.POISON,
    PokemonType.GROUND, PokemonType.FLYING, PokemonType.PSYCHIC, PokemonType.BUG,
    PokemonType.ROCK, PokemonType.GHOST, PokemonType.DRAGON, PokemonType.DARK,
    PokemonType.STEEL, PokemonType.FAIRY,
]

# Climas posibles
WEATHERS = [
    Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM, Weather.HAIL,
    Weather.SNOW, Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM,
]

# Terrenos posibles
FIELDS = [
    Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
]

# Estados de problema
STATUSES = [
    Status.BRN, Status.FRZ, Status.PAR,
    Status.PSN, Status.TOX, Status.SLP,
]

# Boosts posibles en batalla (-6 a +6), normalizados a [-1, 1]
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Numero de movimientos por pokemon
N_MOVES = 4

# Multiplicadores de efectividad de tipo -> indice para one-hot
# x0 (inmune), x0.25, x0.5, x1, x2, x4
EFFECTIVENESS_BUCKETS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

# Tamanio total del vector de observacion (calculado al final del archivo)
OBS_SIZE = None


# ===========================================================================
# Tabla de tipos: type_chart[atacante][defensor] = multiplicador
# ===========================================================================

def _build_type_chart() -> dict:
    """Construye la tabla de efectividades de tipo de Gen 6+."""
    chart = {}
    for t in TYPES:
        chart[t] = {}
        for t2 in TYPES:
            chart[t][t2] = 1.0

    def se(atk, *defs):   # super efectivo
        for d in defs:
            chart[atk][d] = 2.0
    def nve(atk, *defs):  # no muy efectivo
        for d in defs:
            chart[atk][d] = 0.5
    def imm(atk, *defs):  # inmune
        for d in defs:
            chart[atk][d] = 0.0

    T = PokemonType
    se(T.NORMAL)
    nve(T.NORMAL, T.ROCK, T.STEEL)
    imm(T.NORMAL, T.GHOST)

    se(T.FIRE, T.GRASS, T.ICE, T.BUG, T.STEEL)
    nve(T.FIRE, T.FIRE, T.WATER, T.ROCK, T.DRAGON)

    se(T.WATER, T.FIRE, T.GROUND, T.ROCK)
    nve(T.WATER, T.WATER, T.GRASS, T.DRAGON)

    se(T.ELECTRIC, T.WATER, T.FLYING)
    nve(T.ELECTRIC, T.ELECTRIC, T.GRASS, T.DRAGON)
    imm(T.ELECTRIC, T.GROUND)

    se(T.GRASS, T.WATER, T.GROUND, T.ROCK)
    nve(T.GRASS, T.FIRE, T.GRASS, T.POISON, T.FLYING, T.BUG, T.DRAGON, T.STEEL)

    se(T.ICE, T.GRASS, T.GROUND, T.FLYING, T.DRAGON)
    nve(T.ICE, T.FIRE, T.WATER, T.ICE, T.STEEL)

    se(T.FIGHTING, T.NORMAL, T.ICE, T.ROCK, T.DARK, T.STEEL)
    nve(T.FIGHTING, T.POISON, T.FLYING, T.PSYCHIC, T.BUG, T.FAIRY)
    imm(T.FIGHTING, T.GHOST)

    se(T.POISON, T.GRASS, T.FAIRY)
    nve(T.POISON, T.POISON, T.GROUND, T.ROCK, T.GHOST)
    imm(T.POISON, T.STEEL)

    se(T.GROUND, T.FIRE, T.ELECTRIC, T.POISON, T.ROCK, T.STEEL)
    nve(T.GROUND, T.GRASS, T.BUG)
    imm(T.GROUND, T.FLYING)

    se(T.FLYING, T.GRASS, T.FIGHTING, T.BUG)
    nve(T.FLYING, T.ELECTRIC, T.ROCK, T.STEEL)

    se(T.PSYCHIC, T.FIGHTING, T.POISON)
    nve(T.PSYCHIC, T.PSYCHIC, T.STEEL)
    imm(T.PSYCHIC, T.DARK)

    se(T.BUG, T.GRASS, T.PSYCHIC, T.DARK)
    nve(T.BUG, T.FIRE, T.FIGHTING, T.FLYING, T.GHOST, T.STEEL, T.FAIRY)

    se(T.ROCK, T.FIRE, T.ICE, T.FLYING, T.BUG)
    nve(T.ROCK, T.FIGHTING, T.GROUND, T.STEEL)

    se(T.GHOST, T.PSYCHIC, T.GHOST)
    nve(T.GHOST, T.DARK)
    imm(T.GHOST, T.NORMAL)

    se(T.DRAGON, T.DRAGON)
    nve(T.DRAGON, T.STEEL)
    imm(T.DRAGON, T.FAIRY)

    se(T.DARK, T.PSYCHIC, T.GHOST)
    nve(T.DARK, T.FIGHTING, T.DARK, T.FAIRY)

    se(T.STEEL, T.ICE, T.ROCK, T.FAIRY)
    nve(T.STEEL, T.FIRE, T.WATER, T.ELECTRIC, T.STEEL)

    se(T.FAIRY, T.FIGHTING, T.DRAGON, T.DARK)
    nve(T.FAIRY, T.FIRE, T.POISON, T.STEEL)

    return chart


TYPE_CHART = _build_type_chart()


def _type_effectiveness(move_type: PokemonType, defender: Pokemon) -> float:
    """
    Calcula el multiplicador de efectividad de tipo de un movimiento
    contra un pokemon defensor (teniendo en cuenta sus dos tipos).
    """
    mult = 1.0
    for def_type in defender.types:
        if def_type is not None and move_type in TYPE_CHART and def_type in TYPE_CHART[move_type]:
            mult *= TYPE_CHART[move_type][def_type]
    return mult


def _boost_multiplier(boost: int) -> float:
    """Convierte un boost de stat (-6 a +6) en su multiplicador real."""
    if boost >= 0:
        return (2 + boost) / 2.0
    else:
        return 2.0 / (2 - boost)


def _estimate_damage(move: Move, attacker: Pokemon, defender: Pokemon) -> float:
    """
    Estima el dano de un movimiento normalizado a [0, 1] donde 1 = mata al defensor.

    Usa la formula simplificada de dano de Pokemon:
      damage = ((2*level/5 + 2) * power * atk/def) / 50 + 2
    Con level=100 y stats base como aproximacion.

    Retorna fraccion del HP del defensor que se perderia (0 a 1+).
    """
    if move is None or move.base_power == 0:
        return 0.0

    # Determinar si es fisico o especial
    is_physical = move.category.name.lower() == "physical"

    # Stats del atacante
    atk_stat = "atk" if is_physical else "spa"
    atk_base = attacker.base_stats.get(atk_stat, 50)
    atk_boost = _boost_multiplier(attacker.boosts.get(atk_stat, 0))
    atk = atk_base * atk_boost

    # Stats del defensor
    def_stat = "def" if is_physical else "spd"
    def_base = defender.base_stats.get(def_stat, 50)
    def_boost = _boost_multiplier(defender.boosts.get(def_stat, 0))
    def_ = def_base * def_boost

    # Efectividad de tipo
    effectiveness = _type_effectiveness(move.type, defender)
    if effectiveness == 0.0:
        return 0.0

    # STAB
    stab = 1.5 if move.type in (attacker.types or []) else 1.0

    # Formula simplificada (nivel 100)
    power = move.base_power
    raw_damage = ((42 * power * atk / def_) / 50 + 2) * stab * effectiveness

    # Normalizar: HP base del defensor a nivel 100
    # HP = ((2 * base_hp + 31 + 63) * 100 / 100) + 100 + 10  (stats perfectos)
    def_hp_base = defender.base_stats.get("hp", 50)
    def_hp_approx = (2 * def_hp_base + 94) + 110  # aprox con IVs/EVs medios

    damage_fraction = raw_damage / def_hp_approx
    return min(damage_fraction, 2.0) / 2.0  # normalizar a [0, 1]


def _encode_combat_analysis(own: Pokemon, opp: Pokemon, battle: AbstractBattle) -> np.ndarray:
    """
    Analisis de combate: informacion critica que el agente necesita
    para tomar buenas decisiones.

    Por cada movimiento propio (4):
      - Efectividad de tipo (6 bits one-hot: x0, x0.25, x0.5, x1, x2, x4)
      - Dano estimado normalizado (1)
      - Es movimiento de estado (0 dano) (1)
    Total por movimiento: 8
    Total movimientos: 4 x 8 = 32

    Global:
      - Somos mas rapidos que el oponente (1)
      - El oponente nos puede matar de un golpe (estimacion) (1)
      - Podemos matar al oponente con el mejor movimiento (1)
      - Ventaja de tipo general (promedio de efectividades) (1)
    Total global: 4

    Total seccion: 32 + 4 = 36
    """
    parts = []
    moves = list(own.moves.values())

    best_damage = 0.0

    for i in range(N_MOVES):
        move = moves[i] if i < len(moves) else None
        move_vec = np.zeros(8, dtype=np.float32)

        if move is not None and opp is not None:
            effectiveness = _type_effectiveness(move.type, opp)

            # One-hot de efectividad
            bucket_idx = 3  # default x1
            for j, bucket in enumerate(EFFECTIVENESS_BUCKETS):
                if abs(effectiveness - bucket) < 0.01:
                    bucket_idx = j
                    break
            move_vec[bucket_idx] = 1.0

            # Dano estimado
            dmg = _estimate_damage(move, own, opp)
            move_vec[6] = dmg
            best_damage = max(best_damage, dmg)

            # Es movimiento de estado
            move_vec[7] = 1.0 if move.base_power == 0 else 0.0

        parts.append(move_vec)

    # Flags globales
    global_vec = np.zeros(4, dtype=np.float32)

    if opp is not None:
        # Somos mas rapidos
        own_spe = own.base_stats.get("spe", 50) * _boost_multiplier(own.boosts.get("spe", 0))
        opp_spe = opp.base_stats.get("spe", 50) * _boost_multiplier(opp.boosts.get("spe", 0))
        global_vec[0] = 1.0 if own_spe > opp_spe else 0.0

        # El oponente puede KO-earnos (estimacion burda: si tiene alguna amenaza)
        # Usamos el HP restante del propio como proxy
        global_vec[1] = 1.0 if own.current_hp_fraction < 0.3 else 0.0

        # Podemos KO al oponente con el mejor movimiento
        global_vec[2] = 1.0 if best_damage >= opp.current_hp_fraction else 0.0

        # Ventaja de tipo general (promedio de efectividades de nuestros ataques)
        effs = []
        for move in moves:
            if move and move.base_power > 0:
                effs.append(_type_effectiveness(move.type, opp))
        avg_eff = np.mean(effs) if effs else 1.0
        global_vec[3] = min(avg_eff / 4.0, 1.0)  # normalizar (max x4 -> 1.0)

    parts.append(global_vec)
    return np.concatenate(parts)


def _encode_types(pokemon: Pokemon) -> np.ndarray:
    """One-hot de los tipos del pokemon (18 dimensiones)."""
    vec = np.zeros(len(TYPES), dtype=np.float32)
    for t in pokemon.types:
        if t is not None and t in TYPES:
            vec[TYPES.index(t)] = 1.0
    return vec


def _encode_status(pokemon: Pokemon) -> np.ndarray:
    """One-hot del estado del pokemon (6 + 1 para sin estado)."""
    vec = np.zeros(len(STATUSES) + 1, dtype=np.float32)
    if pokemon.status is None:
        vec[-1] = 1.0
    elif pokemon.status in STATUSES:
        vec[STATUSES.index(pokemon.status)] = 1.0
    return vec


def _encode_boosts(pokemon: Pokemon) -> np.ndarray:
    """Boosts normalizados a [-1, 1]."""
    vec = np.zeros(len(BOOST_STATS), dtype=np.float32)
    for i, stat in enumerate(BOOST_STATS):
        boost = pokemon.boosts.get(stat, 0)
        vec[i] = boost / 6.0
    return vec


def _encode_move(move: Move | None, pokemon: Pokemon) -> np.ndarray:
    """
    Codifica un movimiento: tipo(18) + categoria(3) + potencia(1)
    + precision(1) + PP(1) + STAB(1) = 25 bits
    """
    if move is None:
        return np.zeros(25, dtype=np.float32)

    vec = np.zeros(25, dtype=np.float32)

    if move.type in TYPES:
        vec[TYPES.index(move.type)] = 1.0

    category_map = {"physical": 0, "special": 1, "status": 2}
    cat = category_map.get(move.category.name.lower(), 2)
    vec[18 + cat] = 1.0

    base_power = move.base_power or 0
    vec[21] = min(base_power / 250.0, 1.0)

    accuracy = move.accuracy
    vec[22] = 1.0 if accuracy is True else (accuracy / 100.0 if accuracy else 0.0)

    if move.current_pp is not None and move.max_pp:
        vec[23] = move.current_pp / move.max_pp
    else:
        vec[23] = 1.0

    if move.type in (pokemon.types or []):
        vec[24] = 1.0

    return vec


def _encode_active_pokemon(pokemon: Pokemon, is_own: bool) -> np.ndarray:
    """
    Codifica el pokemon activo.
    Propio: info completa. Enemigo: solo lo revelado.
    """
    parts = []

    # HP fraction (1)
    parts.append(np.array([pokemon.current_hp_fraction], dtype=np.float32))

    # Tipos (18)
    parts.append(_encode_types(pokemon))

    # Estado (7)
    parts.append(_encode_status(pokemon))

    # Boosts (7)
    parts.append(_encode_boosts(pokemon))

    if is_own:
        # Stats base normalizados (5)
        stats = pokemon.base_stats
        parts.append(np.array([
            stats.get("atk", 0) / 255.0,
            stats.get("def", 0) / 255.0,
            stats.get("spa", 0) / 255.0,
            stats.get("spd", 0) / 255.0,
            stats.get("spe", 0) / 255.0,
        ], dtype=np.float32))

        # Movimientos (4 x 25 = 100)
        moves = list(pokemon.moves.values())
        for i in range(N_MOVES):
            move = moves[i] if i < len(moves) else None
            parts.append(_encode_move(move, pokemon))
    else:
        parts.append(np.zeros(5, dtype=np.float32))
        parts.append(np.zeros(N_MOVES * 25, dtype=np.float32))

    return np.concatenate(parts)


def _encode_reserve_pokemon(pokemon: Pokemon | None) -> np.ndarray:
    """
    Reserva: HP fraction (1) + tipos (18) + disponible (1) = 20 bits
    """
    if pokemon is None:
        return np.zeros(20, dtype=np.float32)

    parts = []
    parts.append(np.array([pokemon.current_hp_fraction], dtype=np.float32))
    parts.append(_encode_types(pokemon))
    parts.append(np.array([float(not pokemon.fainted)], dtype=np.float32))

    return np.concatenate(parts)


def _encode_field(battle: AbstractBattle) -> np.ndarray:
    """
    Campo: clima(9) + terreno(5) + pantallas propias(3) + pantallas enemigas(3) = 20 bits
    """
    parts = []

    weather_vec = np.zeros(len(WEATHERS) + 1, dtype=np.float32)
    if battle.weather:
        weather_key = list(battle.weather.keys())[0] if battle.weather else None
        if weather_key in WEATHERS:
            weather_vec[WEATHERS.index(weather_key)] = 1.0
        else:
            weather_vec[-1] = 1.0
    else:
        weather_vec[-1] = 1.0
    parts.append(weather_vec)

    field_vec = np.zeros(len(FIELDS) + 1, dtype=np.float32)
    for f in battle.fields:
        if f in FIELDS:
            field_vec[FIELDS.index(f)] = 1.0
            break
    else:
        field_vec[-1] = 1.0
    parts.append(field_vec)

    own_sides = battle.side_conditions
    parts.append(np.array([
        float(SideCondition.REFLECT in own_sides),
        float(SideCondition.LIGHT_SCREEN in own_sides),
        float(SideCondition.AURORA_VEIL in own_sides),
    ], dtype=np.float32))

    opp_sides = battle.opponent_side_conditions
    parts.append(np.array([
        float(SideCondition.REFLECT in opp_sides),
        float(SideCondition.LIGHT_SCREEN in opp_sides),
        float(SideCondition.AURORA_VEIL in opp_sides),
    ], dtype=np.float32))

    return np.concatenate(parts)


def encode_battle(battle: AbstractBattle) -> np.ndarray:
    """
    Punto de entrada principal.
    Devuelve el vector de observacion completo de la batalla.
    """
    parts = []

    own_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # Pokemon activo propio
    if own_active:
        parts.append(_encode_active_pokemon(own_active, is_own=True))
    else:
        parts.append(np.zeros(_active_pokemon_size(), dtype=np.float32))

    # Analisis de combate (NUEVO)
    if own_active and opp_active:
        parts.append(_encode_combat_analysis(own_active, opp_active, battle))
    else:
        parts.append(np.zeros(_combat_analysis_size(), dtype=np.float32))

    # Pokemon en reserva propios (5 slots)
    own_team = [p for p in battle.team.values() if not p.active]
    for i in range(5):
        pokemon = own_team[i] if i < len(own_team) else None
        parts.append(_encode_reserve_pokemon(pokemon))

    # Pokemon activo enemigo
    if opp_active:
        parts.append(_encode_active_pokemon(opp_active, is_own=False))
    else:
        parts.append(np.zeros(_active_pokemon_size(), dtype=np.float32))

    # Pokemon en reserva enemigos (5 slots)
    opp_team = [p for p in battle.opponent_team.values() if not p.active]
    for i in range(5):
        pokemon = opp_team[i] if i < len(opp_team) else None
        parts.append(_encode_reserve_pokemon(pokemon))

    # Condiciones del campo
    parts.append(_encode_field(battle))

    obs = np.concatenate(parts)
    return obs.astype(np.float32)


def _active_pokemon_size() -> int:
    """1 + 18 + 7 + 7 + 5 + 100 = 138"""
    return 1 + 18 + 7 + 7 + 5 + (N_MOVES * 25)


def _combat_analysis_size() -> int:
    """4 movimientos x 8 + 4 globales = 36"""
    return N_MOVES * 8 + 4


def get_observation_size() -> int:
    """Devuelve el tamanio total del vector de observacion."""
    active   = _active_pokemon_size()   # 138
    combat   = _combat_analysis_size()  # 36
    reserve  = 20
    field    = 9 + 5 + 3 + 3           # 20

    total = (
        active +        # activo propio
        combat +        # analisis de combate (NUEVO)
        5 * reserve +   # reserva propia
        active +        # activo enemigo
        5 * reserve +   # reserva enemiga
        field
    )
    return total


OBS_SIZE = get_observation_size()
