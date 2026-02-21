"""
encoder.py
Convierte el estado de una batalla de poke-env en un vector numerico
que el agente RL puede procesar.

Estructura del vector de observacion:
  - Pokemon activo propio (HP, stats, tipos, habilidad, estado, boosts)
  - Pokemon en reserva propios x5 (HP, tipos, disponible)
  - Pokemon activo enemigo (HP visible, tipos, habilidad, estado, boosts)
  - Pokemon en reserva enemigos x5 (HP visible, tipos)
  - Condiciones del campo (clima, terreno, pantallas, etc.)
"""

import numpy as np
from poke_env.environment import AbstractBattle, Pokemon, Move, Weather, Field, SideCondition, Status, PokemonType

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

# Tamanio total del vector de observacion (calculado al final del archivo)
OBS_SIZE = None


def _encode_types(pokemon: Pokemon) -> np.ndarray:
    """One-hot de los tipos del pokemon (18 dimensiones x2 slots = 18 bits)."""
    vec = np.zeros(len(TYPES), dtype=np.float32)
    for t in pokemon.types:
        if t is not None and t in TYPES:
            vec[TYPES.index(t)] = 1.0
    return vec


def _encode_status(pokemon: Pokemon) -> np.ndarray:
    """One-hot del estado del pokemon (6 dimensiones + 1 para sin estado)."""
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
        vec[i] = boost / 6.0  # normalizar de [-6,6] a [-1,1]
    return vec


def _encode_move(move: Move | None, pokemon: Pokemon) -> np.ndarray:
    """
    Codifica un movimiento en un vector:
    - tipo (18 bits one-hot)
    - categoria (3 bits: fisico, especial, estado)
    - potencia normalizada (1)
    - precision normalizada (1)
    - PP restantes normalizados (1)
    - STAB (1)
    Total: 25 bits por movimiento
    """
    if move is None:
        return np.zeros(25, dtype=np.float32)

    vec = np.zeros(25, dtype=np.float32)

    # Tipo del movimiento
    if move.type in TYPES:
        vec[TYPES.index(move.type)] = 1.0

    # Categoria
    category_map = {"physical": 0, "special": 1, "status": 2}
    cat = category_map.get(move.category.name.lower(), 2)
    vec[18 + cat] = 1.0

    # Potencia normalizada (base 0-250 → 0-1)
    base_power = move.base_power or 0
    vec[21] = min(base_power / 250.0, 1.0)

    # Precision normalizada (0-100 → 0-1, None = always hits = 1)
    accuracy = move.accuracy
    vec[22] = 1.0 if accuracy is True else (accuracy / 100.0 if accuracy else 0.0)

    # PP restantes normalizados
    if move.current_pp is not None and move.max_pp:
        vec[23] = move.current_pp / move.max_pp
    else:
        vec[23] = 1.0

    # STAB
    if move.type in (pokemon.types or []):
        vec[24] = 1.0

    return vec


def _encode_active_pokemon(pokemon: Pokemon, is_own: bool) -> np.ndarray:
    """
    Codifica el pokemon activo.
    Para el propio tenemos info completa.
    Para el enemigo solo lo que se ha revelado.
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
        # Stats base normalizados (atk, def, spa, spd, spe) - (5)
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
        # Para el enemigo, stats y movimientos no son completamente conocidos
        # Usamos zeros del mismo tamanio para mantener el vector consistente
        parts.append(np.zeros(5, dtype=np.float32))
        parts.append(np.zeros(N_MOVES * 25, dtype=np.float32))

    return np.concatenate(parts)


def _encode_reserve_pokemon(pokemon: Pokemon | None) -> np.ndarray:
    """
    Codifica un pokemon en reserva (propio o enemigo).
    Info reducida: HP fraction, tipos, si esta disponible.
    Total: 1 + 18 + 1 = 20 bits
    """
    if pokemon is None:
        return np.zeros(20, dtype=np.float32)

    parts = []
    parts.append(np.array([pokemon.current_hp_fraction], dtype=np.float32))
    parts.append(_encode_types(pokemon))
    # Disponible (no fainted, no activo)
    available = float(not pokemon.fainted)
    parts.append(np.array([available], dtype=np.float32))

    return np.concatenate(parts)


def _encode_field(battle: AbstractBattle) -> np.ndarray:
    """
    Codifica las condiciones del campo:
    - Clima (8 bits one-hot + 1 sin clima)
    - Terreno (4 bits one-hot + 1 sin terreno)
    - Pantallas lado propio: reflect, light_screen, aurora_veil (3)
    - Pantallas lado enemigo: reflect, light_screen, aurora_veil (3)
    Total: 9 + 5 + 3 + 3 = 20 bits
    """
    parts = []

    # Clima
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

    # Terreno
    field_vec = np.zeros(len(FIELDS) + 1, dtype=np.float32)
    for f in battle.fields:
        if f in FIELDS:
            field_vec[FIELDS.index(f)] = 1.0
            break
    else:
        field_vec[-1] = 1.0
    parts.append(field_vec)

    # Pantallas propias
    own_sides = battle.side_conditions
    parts.append(np.array([
        float(SideCondition.REFLECT in own_sides),
        float(SideCondition.LIGHT_SCREEN in own_sides),
        float(SideCondition.AURORA_VEIL in own_sides),
    ], dtype=np.float32))

    # Pantallas enemigas
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

    # Pokemon activo propio
    own_active = battle.active_pokemon
    if own_active:
        parts.append(_encode_active_pokemon(own_active, is_own=True))
    else:
        # vector de zeros del tamanio correcto si no hay activo (no deberia pasar)
        parts.append(np.zeros(_active_pokemon_size(), dtype=np.float32))

    # Pokemon en reserva propios (5 slots)
    own_team = [p for p in battle.team.values() if not p.active]
    for i in range(5):
        pokemon = own_team[i] if i < len(own_team) else None
        parts.append(_encode_reserve_pokemon(pokemon))

    # Pokemon activo enemigo
    opp_active = battle.opponent_active_pokemon
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
    """Tamanio del vector de un pokemon activo: 1+18+7+7+5+100 = 138."""
    return 1 + 18 + 7 + 7 + 5 + (N_MOVES * 25)


def get_observation_size() -> int:
    """Devuelve el tamanio total del vector de observacion."""
    active = _active_pokemon_size()   # 138
    reserve = 20                       # por pokemon en reserva
    field = 9 + 5 + 3 + 3             # 20

    total = (
        active +       # activo propio
        5 * reserve +  # reserva propia
        active +       # activo enemigo
        5 * reserve +  # reserva enemiga
        field
    )
    return total


OBS_SIZE = get_observation_size()
