"""
encoder.py
Convierte el estado de una batalla de poke-env en un vector numerico
que el agente RL puede procesar.

=============================================================================
FORMULAS Y CALCULOS USADOS
=============================================================================

1. FORMULA DE DAÑO (Gen 6 / 7 / 8 / 9 — identica en todas):

   damage = floor(floor(floor(2*level/5 + 2) * power * atk/def / 50) + 2)
            * targets * weather * badge * critical * random * STAB
            * type1 * type2 * burn * other * zpower

   Donde (nivel 100, singles competitivo):
     floor(2*100/5 + 2) = 42  (constante)
     power    = potencia base del movimiento
     atk      = stat de Ataque o Ataque Especial del atacante (con boosts)
     def      = stat de Defensa o Defensa Especial del defensor (con boosts)
     targets  = 1.0 (singles; 0.75 si multiples objetivos en dobles)
     weather  = 1.5 / 0.5 segun tipo y clima activo
     badge    = 1.0 (no se usa en competitivo moderno)
     critical = 1.0 (promedio; 1.5 en critico real)
     random   = 0.925 (promedio entre 0.85-1.00; ver _ko_probability)
     STAB     = 1.5 (mismo tipo) | 2.0 (Adaptability) | siempre 1.5 (Protean)
     type1    = efectividad vs tipo 1 del defensor (0, 0.5, 1, 2)
     type2    = efectividad vs tipo 2 del defensor (idem)
     burn     = 0.5 si el atacante esta quemado y el movimiento es fisico
     other    = items + habilidades + pantallas + terreno (ver items_and_abilities.py)
     zpower   = 1.0 (sin z-moves en el calculo estandar)

2. ESTIMACION DE STATS (nivel 100, IVs 31, EVs 85 ~ 252/3):
   stat = floor((2 * base + 31 + 21) * 100 / 100) + 5   (no HP)
   HP   = floor((2 * base + 31 + 21) * 100 / 100) + 110

3. TABLA DE TIPOS: Gen 6+ (18 tipos, Fairy incluido).

4. ITEMS Y HABILIDADES: ver src/state/items_and_abilities.py
   para la lista completa con efectos.

5. DETECCION DE KO (probabilistica):
   Calcula dmg con roll=0.85 (minimo) y roll=1.0 (maximo).
   Interpola la probabilidad de KO segun donde cae el HP actual
   en el rango [dmg_min, dmg_max].

=============================================================================

Estructura del vector de observacion (570 dims):
  1. Pokemon activo propio  (138): HP + tipos + estado + boosts + stats + moves
  2. Analisis de combate    ( 36): 4 moves x 8 dims + 4 globales
  3. Analisis de cambios    ( 30): 5 reservas x 6 dims
  4. Reserva propia x5      (100): 5 x (HP + tipos + disponible)
  5. Pokemon activo enemigo (138)
  6. Reserva enemiga x5     (100)
  7. Campo                  ( 28): clima + terreno + pantallas + hazards

TOTAL: 138 + 36 + 30 + 100 + 138 + 100 + 28 = 570
"""

import math
import numpy as np
from poke_env.battle import (
    AbstractBattle, Pokemon, Move, Weather, Field, SideCondition, Status, PokemonType
)

# Importar tablas de items y habilidades
from src.state.items_and_abilities import (
    clean,
    get_attack_item_mult,
    get_defense_item_divisor,
    get_speed_item_mult,
    get_speed_ability_mult,
    get_attack_ability_params,
    get_defense_ability_params,
    TYPE_BOOST_ITEMS,
    PUNCH_MOVES,
    BITE_MOVES,
    RECOIL_MOVES,
    PULSE_MOVES,
    SOUND_MOVES,
    CONTACT_MOVES,
    BALL_BOMB_MOVES,
)

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

# Tamanio total (calculado al final)
OBS_SIZE = None


# ===========================================================================
# Tabla de tipos Gen 6+
# ===========================================================================

def _build_type_chart() -> dict:
    """Construye la tabla de efectividades de tipo de Gen 6+ (18 tipos, Fairy incluido)."""
    chart = {}
    for t in TYPES:
        chart[t] = {}
        for t2 in TYPES:
            chart[t][t2] = 1.0

    def se(atk, *defs):
        for d in defs: chart[atk][d] = 2.0
    def nve(atk, *defs):
        for d in defs: chart[atk][d] = 0.5
    def imm(atk, *defs):
        for d in defs: chart[atk][d] = 0.0

    T = PokemonType
    nve(T.NORMAL, T.ROCK, T.STEEL);                                     imm(T.NORMAL, T.GHOST)
    se(T.FIRE, T.GRASS, T.ICE, T.BUG, T.STEEL);     nve(T.FIRE, T.FIRE, T.WATER, T.ROCK, T.DRAGON)
    se(T.WATER, T.FIRE, T.GROUND, T.ROCK);           nve(T.WATER, T.WATER, T.GRASS, T.DRAGON)
    se(T.ELECTRIC, T.WATER, T.FLYING);               nve(T.ELECTRIC, T.ELECTRIC, T.GRASS, T.DRAGON);  imm(T.ELECTRIC, T.GROUND)
    se(T.GRASS, T.WATER, T.GROUND, T.ROCK);          nve(T.GRASS, T.FIRE, T.GRASS, T.POISON, T.FLYING, T.BUG, T.DRAGON, T.STEEL)
    se(T.ICE, T.GRASS, T.GROUND, T.FLYING, T.DRAGON); nve(T.ICE, T.FIRE, T.WATER, T.ICE, T.STEEL)
    se(T.FIGHTING, T.NORMAL, T.ICE, T.ROCK, T.DARK, T.STEEL); nve(T.FIGHTING, T.POISON, T.FLYING, T.PSYCHIC, T.BUG, T.FAIRY); imm(T.FIGHTING, T.GHOST)
    se(T.POISON, T.GRASS, T.FAIRY);                  nve(T.POISON, T.POISON, T.GROUND, T.ROCK, T.GHOST); imm(T.POISON, T.STEEL)
    se(T.GROUND, T.FIRE, T.ELECTRIC, T.POISON, T.ROCK, T.STEEL); nve(T.GROUND, T.GRASS, T.BUG);    imm(T.GROUND, T.FLYING)
    se(T.FLYING, T.GRASS, T.FIGHTING, T.BUG);        nve(T.FLYING, T.ELECTRIC, T.ROCK, T.STEEL)
    se(T.PSYCHIC, T.FIGHTING, T.POISON);             nve(T.PSYCHIC, T.PSYCHIC, T.STEEL);              imm(T.PSYCHIC, T.DARK)
    se(T.BUG, T.GRASS, T.PSYCHIC, T.DARK);          nve(T.BUG, T.FIRE, T.FIGHTING, T.FLYING, T.GHOST, T.STEEL, T.FAIRY)
    se(T.ROCK, T.FIRE, T.ICE, T.FLYING, T.BUG);     nve(T.ROCK, T.FIGHTING, T.GROUND, T.STEEL)
    se(T.GHOST, T.PSYCHIC, T.GHOST);                 nve(T.GHOST, T.DARK);                             imm(T.GHOST, T.NORMAL)
    se(T.DRAGON, T.DRAGON);                          nve(T.DRAGON, T.STEEL);                           imm(T.DRAGON, T.FAIRY)
    se(T.DARK, T.PSYCHIC, T.GHOST);                  nve(T.DARK, T.FIGHTING, T.DARK, T.FAIRY)
    se(T.STEEL, T.ICE, T.ROCK, T.FAIRY);             nve(T.STEEL, T.FIRE, T.WATER, T.ELECTRIC, T.STEEL)
    se(T.FAIRY, T.FIGHTING, T.DRAGON, T.DARK);       nve(T.FAIRY, T.FIRE, T.POISON, T.STEEL)
    return chart


TYPE_CHART = _build_type_chart()


def _type_effectiveness(move_type: PokemonType, defender: Pokemon) -> float:
    """
    Calcula el multiplicador de efectividad de tipo contra el defensor
    (product de los dos tipos del defensor).
    """
    mult = 1.0
    for def_type in defender.types:
        if def_type is not None and move_type in TYPE_CHART and def_type in TYPE_CHART[move_type]:
            mult *= TYPE_CHART[move_type][def_type]
    return mult


# ===========================================================================
# Helpers de stats y boosts
# ===========================================================================

def _boost_multiplier(boost: int) -> float:
    """Convierte un boost de stat (-6 a +6) en su multiplicador real."""
    if boost >= 0:
        return (2 + boost) / 2.0
    else:
        return 2.0 / (2 - boost)


def _stat_at_100(base: int, is_hp: bool = False) -> float:
    """
    Estima el stat a nivel 100 con IVs 31 y EVs 85 (252/3 aprox).
    stat = floor((2*base + 31 + 21) * 100/100) + 5    (no HP)
    HP   = floor((2*base + 31 + 21) * 100/100) + 110
    """
    base_val = math.floor((2 * base + 31 + 21) * 100 / 100)
    return float(base_val + (110 if is_hp else 5))


# ===========================================================================
# Multiplicadores de clima y terreno
# ===========================================================================

def _get_weather_key(battle):
    """Devuelve el enum del clima activo o None."""
    if battle is None or not battle.weather:
        return None
    return list(battle.weather.keys())[0]


def _get_weather_multiplier(move_type: PokemonType, battle) -> float:
    """
    Sol (SUNNYDAY/DESOLATELAND):   Fire x1.5, Water x0.5
    Lluvia (RAINDANCE/PRIMORDIALSEA): Water x1.5, Fire x0.5
    Arena/Granizo/Nieve: sin efecto en daño directo.
    """
    w = _get_weather_key(battle)
    if w is None:
        return 1.0
    sunny = {Weather.SUNNYDAY, Weather.DESOLATELAND}
    rainy = {Weather.RAINDANCE, Weather.PRIMORDIALSEA}
    if w in sunny:
        if move_type == PokemonType.FIRE:  return 1.5
        if move_type == PokemonType.WATER: return 0.5
    elif w in rainy:
        if move_type == PokemonType.WATER: return 1.5
        if move_type == PokemonType.FIRE:  return 0.5
    return 1.0


def _get_terrain_multiplier(move_type: PokemonType, battle) -> float:
    """
    Electric Terrain: Electric x1.3
    Grassy Terrain:   Grass x1.3
    Misty Terrain:    Dragon x0.5
    Psychic Terrain:  Psychic x1.3
    Solo aplica a pokemon en tierra (asumimos que si).
    """
    if battle is None or not battle.fields:
        return 1.0
    if Field.ELECTRIC_TERRAIN in battle.fields and move_type == PokemonType.ELECTRIC: return 1.3
    if Field.GRASSY_TERRAIN   in battle.fields and move_type == PokemonType.GRASS:    return 1.3
    if Field.MISTY_TERRAIN    in battle.fields and move_type == PokemonType.DRAGON:   return 0.5
    if Field.PSYCHIC_TERRAIN  in battle.fields and move_type == PokemonType.PSYCHIC:  return 1.3
    return 1.0


def _get_screen_divisor(battle, is_physical: bool, is_own_attack: bool) -> float:
    """
    Reflect / Light Screen / Aurora Veil: divide el daño por 2 en singles.
    is_own_attack=True  => las pantallas del OPONENTE reducen nuestro daño
    is_own_attack=False => las pantallas PROPIAS reducen el daño recibido
    """
    if battle is None:
        return 1.0
    sides = battle.opponent_side_conditions if is_own_attack else battle.side_conditions
    if SideCondition.AURORA_VEIL in sides:
        return 2.0
    if is_physical  and SideCondition.REFLECT      in sides: return 2.0
    if not is_physical and SideCondition.LIGHT_SCREEN in sides: return 2.0
    return 1.0


# ===========================================================================
# Multiplicadores de habilidad — usando items_and_abilities.py
# ===========================================================================

def _get_ability_multipliers(attacker: Pokemon, defender: Pokemon, move: Move):
    """
    Retorna (atk_mult, def_mult, stab_override, immune).

    atk_mult     : multiplicador de daño del atacante (habilidad)
    def_mult     : DIVISOR de daño del defensor (habilidad)
    stab_override: float si reemplaza STAB, None si usa el 1.5 estandar
    immune       : True si la habilidad del defensor lo hace inmune

    Usa los diccionarios de items_and_abilities.py como referencia.
    """
    atk_ability = clean(getattr(attacker, 'ability', None))
    def_ability = clean(getattr(defender, 'ability', None))

    move_type   = move.type if move is not None else None
    is_physical = move.category.name.lower() == "physical" if move is not None else True
    base_power  = move.base_power if move is not None else 0
    move_name   = move.id.lower() if move is not None else ""

    effectiveness = _type_effectiveness(move_type, defender) if move_type is not None else 1.0
    is_supereff   = effectiveness >= 2.0

    atk_mult     = 1.0
    stab_override = None
    immune        = False

    # -------------------------------------------------------
    # Habilidades del ATACANTE
    # -------------------------------------------------------
    if atk_ability:
        params = get_attack_ability_params(atk_ability)

        # STAB overrides
        if atk_ability == "adaptability":
            stab_override = 2.0

        elif atk_ability in ("protean", "libero"):
            stab_override = 1.5   # siempre STAB para cualquier movimiento

        # Multiplicadores directos de daño
        elif atk_ability in ("steelworker", "transistor", "dragonsmaw", "rockypayload"):
            p_type = params.get("type")
            if move_type == p_type:
                atk_mult *= params.get("mult", 1.0)

        elif atk_ability == "hadronengine":
            if move_type == PokemonType.ELECTRIC:
                atk_mult *= 1.333  # Electric x4/3 en Electric Terrain (simplificado: siempre)

        elif atk_ability == "orichalcumpulse":
            if move_type == PokemonType.FIRE:
                atk_mult *= 1.333  # Fire x4/3 en sol (simplificado: siempre)

        elif atk_ability == "hustle" and is_physical:
            atk_mult *= 1.5

        elif atk_ability == "gorillatactics" and is_physical:
            atk_mult *= 1.5

        elif atk_ability == "guts" and is_physical:
            if attacker.status is not None:
                atk_mult *= 1.5   # burn penalty se anula dentro de _estimate_damage

        elif atk_ability == "flareboost" and not is_physical:
            if attacker.status == Status.BRN:
                atk_mult *= 1.5

        elif atk_ability == "toxicboost" and is_physical:
            if attacker.status in (Status.PSN, Status.TOX):
                atk_mult *= 1.5

        elif atk_ability == "blaze" and move_type == PokemonType.FIRE:
            if attacker.current_hp_fraction < 0.33:
                atk_mult *= 1.5

        elif atk_ability == "torrent" and move_type == PokemonType.WATER:
            if attacker.current_hp_fraction < 0.33:
                atk_mult *= 1.5

        elif atk_ability == "overgrow" and move_type == PokemonType.GRASS:
            if attacker.current_hp_fraction < 0.33:
                atk_mult *= 1.5

        elif atk_ability == "swarm" and move_type == PokemonType.BUG:
            if attacker.current_hp_fraction < 0.33:
                atk_mult *= 1.5

        elif atk_ability == "technician":
            if 0 < base_power <= 60:
                atk_mult *= 1.5

        elif atk_ability == "sheerforce":
            if base_power > 0:
                atk_mult *= 1.3

        elif atk_ability == "reckless":
            if move_name in RECOIL_MOVES:
                atk_mult *= 1.2

        elif atk_ability == "ironfist":
            if move_name in PUNCH_MOVES:
                atk_mult *= 1.2

        elif atk_ability == "strongjaw":
            if move_name in BITE_MOVES:
                atk_mult *= 1.5

        elif atk_ability == "megalauncher":
            if move_name in PULSE_MOVES:
                atk_mult *= 1.5

        elif atk_ability == "toughclaws":
            if move_name in CONTACT_MOVES:
                atk_mult *= 1.3

        elif atk_ability == "punkrock":
            if move_name in SOUND_MOVES:
                atk_mult *= 1.3

        elif atk_ability == "sandforce" and is_physical:
            if move_type in (PokemonType.ROCK, PokemonType.STEEL, PokemonType.GROUND):
                atk_mult *= 1.3

        elif atk_ability == "analytic":
            # simplificacion: asumimos que el rival actua primero si es mas rapido
            own_spe = _stat_at_100(attacker.base_stats.get("spe", 50))
            opp_spe = _stat_at_100(defender.base_stats.get("spe", 50))
            if opp_spe > own_spe:
                atk_mult *= 1.3

        elif atk_ability == "tintedlens":
            # NVE se trata como x1 efectivo
            if effectiveness < 1.0:
                atk_mult *= (1.0 / effectiveness) if effectiveness > 0 else 1.0

        elif atk_ability == "neuroforce" and is_supereff:
            atk_mult *= 1.25

        elif atk_ability in ("aerilate", "pixilate", "refrigerate", "galvanize"):
            if move_type == PokemonType.NORMAL:
                atk_mult *= 1.2  # Normal -> tipo convertido con x1.2

        elif atk_ability == "solarpower" and not is_physical:
            w = _get_weather_key(None)   # sin battle aqui; se aplica siempre como aprox
            atk_mult *= 1.5

        elif atk_ability == "punchingglove" and move_name in PUNCH_MOVES and is_physical:
            atk_mult *= 1.1   # Punch Glove es item, pero hay habilidades similares

        elif atk_ability == "supremeoverlord":
            # +10% por cada aliado caido (max 5).
            # No tenemos acceso directo al equipo desde el Pokemon, usamos 0 como estimacion segura.
            atk_mult *= 1.0  # simplificacion conservadora

    # -------------------------------------------------------
    # Habilidades del DEFENSOR
    # -------------------------------------------------------
    def_mult = 1.0

    if def_ability and move_type is not None:
        params = get_defense_ability_params(def_ability)
        immune_val = params.get("immune")

        # Inmunidades de tipo
        if isinstance(immune_val, PokemonType) and move_type == immune_val:
            immune = True

        elif def_ability == "dryskin" and move_type == PokemonType.WATER:
            immune = True

        elif def_ability == "wonderguard" and not is_supereff:
            immune = True

        elif def_ability == "bulletproof" and move_name in BALL_BOMB_MOVES:
            immune = True

        elif def_ability == "soundproof" and move_name in SOUND_MOVES:
            immune = True

        # Reducciones de daño
        elif def_ability in ("thickfat",):
            itype = params.get("type", set())
            if isinstance(itype, set):
                if move_type in itype:
                    def_mult *= params.get("divisor", 1.0)
            elif isinstance(itype, PokemonType):
                if move_type == itype:
                    def_mult *= params.get("divisor", 1.0)

        elif def_ability in ("heatproof", "purifyingsalt"):
            itype = params.get("type")
            if move_type == itype:
                def_mult *= params.get("divisor", 1.0)

        elif def_ability == "icescales" and not is_physical:
            def_mult *= params.get("divisor", 2.0)

        elif def_ability in ("multiscale", "shadowshield"):
            if defender.current_hp_fraction >= 0.99:
                def_mult *= params.get("divisor", 2.0)

        elif def_ability in ("filter", "solidrock", "prismarmor") and is_supereff:
            def_mult *= params.get("divisor", 1.333)

        elif def_ability == "fluffy":
            if move_type == PokemonType.FIRE:
                def_mult *= 0.5   # Fire hace x2 daño -> divisor 0.5 (aumenta daño)
            elif is_physical and move_name in CONTACT_MOVES:
                def_mult *= params.get("divisor", 2.0)   # contacto x0.5

        # Fur Coat es habilidad, AUMENTA defensa fisica efectiva x2
        elif def_ability == "furcoat" and is_physical:
            def_mult *= params.get("divisor", 2.0)

        elif def_ability == "marvelscale" and is_physical:
            if defender.status is not None:
                def_mult *= params.get("divisor", 1.5)

        elif def_ability == "grasspelt" and is_physical:
            if hasattr(defender, '_battle') and defender._battle and Field.GRASSY_TERRAIN in defender._battle.fields:
                def_mult *= 1.5

        elif def_ability == "punkrock" and move_name in SOUND_MOVES:
            def_mult *= params.get("divisor", 2.0)

        elif def_ability == "dryskin" and move_type == PokemonType.FIRE:
            def_mult *= 0.8   # recibe 25% mas daño de fuego (divisor < 1 = mas daño)

    return atk_mult, def_mult, stab_override, immune


# ===========================================================================
# Formula de daño principal
# ===========================================================================

def _estimate_damage(
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    battle=None,
    is_own_attack: bool = True,
    random_roll: float = 0.925,
) -> float:
    """
    Estima el daño de un movimiento normalizado a [0, 1] donde 1 = mata al defensor.

    FORMULA Gen 6/7/8/9 (nivel 100):
    =================================
      base   = floor(floor(42 * power * atk/def / 50) + 2)
      damage = base * 1.0 * weather * 1.0 * 1.0 * random_roll
                    * STAB * type1 * type2 * burn * other * 1.0
                                                (targets/badge/critical/zpower = 1.0)
      other  = item_atk * ability_atk * terrain / (ability_def * item_def * screen)

    Retorna fraccion del HP total del defensor (0 a 1, clippeado).
    La comparacion correcta para KO:
        _estimate_damage(...) >= defender.current_hp_fraction
    """
    if move is None or move.base_power == 0:
        return 0.0

    is_physical = move.category.name.lower() == "physical"
    move_type   = move.type
    move_name   = move.id.lower() if move.id else ""

    # --- Habilidades ---
    atk_ability_mult, def_ability_mult, stab_override, immune = _get_ability_multipliers(
        attacker, defender, move
    )
    if immune:
        return 0.0

    # --- Stats del atacante (con boosts) ---
    atk_stat_key = "atk" if is_physical else "spa"
    atk_base  = attacker.base_stats.get(atk_stat_key, 50)
    atk_boost = _boost_multiplier(attacker.boosts.get(atk_stat_key, 0))
    atk_real  = _stat_at_100(atk_base) * atk_boost

    # --- Stats del defensor (con boosts) ---
    def_stat_key = "def" if is_physical else "spd"
    def_base  = defender.base_stats.get(def_stat_key, 50)
    def_boost = _boost_multiplier(defender.boosts.get(def_stat_key, 0))
    def_real  = _stat_at_100(def_base) * def_boost

    # --- Efectividad de tipo ---
    effectiveness = _type_effectiveness(move_type, defender)
    if effectiveness == 0.0:
        return 0.0

    # --- STAB ---
    attacker_types = attacker.types or []
    if stab_override is not None:
        # Protean/Libero: siempre STAB | Adaptability: STAB x2 si mismo tipo
        stab = stab_override if (stab_override == 1.5 or move_type in attacker_types) else 1.0
    else:
        stab = 1.5 if move_type in attacker_types else 1.0

    # --- Item del atacante ---
    atk_item = clean(getattr(attacker, 'item', None))
    item_atk_mult = get_attack_item_mult(atk_item, is_physical, move_type, effectiveness)

    # --- Item del defensor ---
    def_item = clean(getattr(defender, 'item', None))
    item_def_div  = get_defense_item_divisor(def_item, is_physical, move_type)

    # --- Clima ---
    weather_mult = _get_weather_multiplier(move_type, battle)

    # --- Terreno ---
    terrain_mult = _get_terrain_multiplier(move_type, battle)

    # --- Pantallas ---
    screen_div = _get_screen_divisor(battle, is_physical, is_own_attack)

    # --- Burn: x0.5 si quemado y fisico (Guts lo anula — ya aplicado en atk_ability_mult) ---
    burn_mult = 1.0
    guts_active = clean(getattr(attacker, 'ability', None)) == "guts"
    if is_physical and attacker.status == Status.BRN and not guts_active:
        burn_mult = 0.5

    # =========================================================
    # FORMULA GEN 6+ (nivel 100):
    #   floor(floor(floor(2*100/5 + 2) * power * atk/def / 50) + 2)
    #   * targets(1) * weather * badge(1) * critical(1) * random
    #   * STAB * type1*type2 * burn * other * zpower(1)
    # =========================================================
    power = move.base_power

    # Paso 1: base entera con floors (tal cual la formula oficial)
    base_damage = math.floor(
        math.floor(math.floor(2 * 100 / 5 + 2) * power * atk_real / def_real / 50) + 2
    )

    # Paso 2: multiplicadores en cadena
    damage = (
        base_damage
        # targets=1, badge=1, critical=1, zpower=1
        * weather_mult
        * random_roll
        * stab
        * effectiveness         # type1 * type2 ya combinados
        * burn_mult
        # "other": items + habilidades + terreno + pantallas
        * item_atk_mult
        * atk_ability_mult
        * terrain_mult
        / def_ability_mult      # divisores de habilidad defensiva
        / item_def_div          # divisores de item defensivo
        / screen_div            # divisores de pantalla
    )

    # HP del defensor a nivel 100
    def_hp_base   = defender.base_stats.get("hp", 50)
    def_hp_approx = _stat_at_100(def_hp_base, is_hp=True)

    return float(min(max(damage / def_hp_approx, 0.0), 1.0))


# ===========================================================================
# KO probabilistico
# ===========================================================================

def _best_damage_vs(
    attacker: Pokemon,
    defender: Pokemon,
    battle=None,
    is_own_attack: bool = True,
    roll: float = 0.925,
) -> float:
    """
    Daño maximo que el atacante puede hacer al defensor con su mejor movimiento.
    roll controla el factor aleatorio: 0.85 = minimo, 0.925 = promedio, 1.0 = maximo.

    Si no tiene movimientos conocidos, estima con tipos STAB y potencia 80
    usando el mayor stat ofensivo (atk o spa).
    """
    moves = list(attacker.moves.values()) if attacker.moves else []
    best = 0.0

    if moves:
        for mv in moves:
            if mv.base_power > 0:
                d = _estimate_damage(mv, attacker, defender,
                                     battle=battle, is_own_attack=is_own_attack, random_roll=roll)
                best = max(best, d)
    else:
        # Sin movimientos conocidos: estimacion con STAB y potencia 80
        atk_base = max(attacker.base_stats.get("atk", 80), attacker.base_stats.get("spa", 80))
        def_base = min(defender.base_stats.get("def", 80), defender.base_stats.get("spd", 80))
        atk_real = _stat_at_100(atk_base)
        def_real = _stat_at_100(def_base)
        def_hp   = _stat_at_100(defender.base_stats.get("hp", 80), is_hp=True)

        for t in attacker.types:
            if t is None:
                continue
            eff = _type_effectiveness(t, defender)
            if eff == 0.0:
                continue
            base_dmg = math.floor(
                math.floor(math.floor(2 * 100 / 5 + 2) * 80 * atk_real / def_real / 50) + 2
            )
            dmg = base_dmg * roll * 1.5 * eff   # STAB asumido
            best = max(best, min(dmg / def_hp, 1.0))

    return best


def _ko_probability(attacker: Pokemon, defender: Pokemon, battle=None, is_own_attack: bool = True) -> float:
    """
    Probabilidad de KO en 1 golpe como valor continuo [0, 1].

    - dmg_min (roll=0.85) >= HP actual  -> 1.0 (KO garantizado)
    - dmg_max (roll=1.0)  <  HP actual  -> 0.0 (imposible)
    - entre medias: interpolacion lineal segun cuanto del rango supera HP.

    Nota: HP actual y daño son ambos fracciones del HP TOTAL del defensor.
    """
    dmg_min = _best_damage_vs(attacker, defender, battle, is_own_attack, roll=0.85)
    dmg_max = _best_damage_vs(attacker, defender, battle, is_own_attack, roll=1.00)
    hp_curr = defender.current_hp_fraction

    if dmg_min >= hp_curr:
        return 1.0
    if dmg_max < hp_curr:
        return 0.0

    damage_range = dmg_max - dmg_min
    if damage_range < 1e-8:
        return 0.0
    prob = (dmg_max - hp_curr) / damage_range
    return float(min(max(prob, 0.0), 1.0))


# ===========================================================================
# Velocidad real (con Choice Scarf y habilidades de clima)
# ===========================================================================

def _real_speed(pokemon: Pokemon, battle=None) -> float:
    """
    Velocidad real estimada del pokemon (fraccion del speed nominal).
    Considera en orden:
      1. stat base a nivel 100 (IVs 31, EVs 85)
      2. boosts de batalla (-6 a +6)
      3. paralisis: x0.5 (excepto Quick Feet)
      4. item de velocidad (Choice Scarf x1.5, Iron Ball x0.5, etc.)
      5. habilidad de velocidad condicional al clima/terreno/estado
         (Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Surge Surfer,
          Quick Feet, Unburden, Speed Boost, Slow Start, etc.)
    """
    spe_base  = pokemon.base_stats.get("spe", 50)
    spe_boost = _boost_multiplier(pokemon.boosts.get("spe", 0))
    spe_real  = _stat_at_100(spe_base) * spe_boost

    # Paralisis: x0.5 de velocidad (Quick Feet lo anula y hace x1.5)
    ability = clean(getattr(pokemon, 'ability', None))
    if pokemon.status == Status.PAR and ability != "quickfeet":
        spe_real *= 0.5

    # Item de velocidad (get_speed_item_mult devuelve 1.5 para scarf, etc.)
    item = clean(getattr(pokemon, 'item', None))
    spe_real *= get_speed_item_mult(item)

    # Habilidad de velocidad — pasar sets de enums correctos
    if ability:
        # battle.weather es dict {WeatherEnum: turns}, necesitamos el set de enums
        weather_set = set(battle.weather.keys()) if battle and battle.weather else set()
        # battle.fields es dict {FieldEnum: turns}
        field_set   = set(battle.fields.keys())  if battle and battle.fields  else set()
        has_status  = pokemon.status is not None
        spe_real *= get_speed_ability_mult(
            ability,
            weather=weather_set,
            field=field_set,
            has_status=has_status,
        )

    return spe_real


# ===========================================================================
# Analisis de cambios (switch analysis)
# ===========================================================================

def _encode_switch_analysis(reserve_pokemon: list, opp_active: Pokemon, battle=None) -> np.ndarray:
    """
    Por cada pokemon en reserva (hasta 5) devuelve 6 dims:
      [0] Mejor daño ofensivo vs oponente (formula Gen 6+, normalizado 0-1)
      [1] Resiste el tipo principal del oponente (1.0 si mult <= 0.5)
      [2] Inmune al tipo principal del oponente (1.0 si mult == 0.0)
      [3] Ventaja de velocidad vs oponente (1.0 si mas rapido, con Scarf)
      [4] HP fraction del pokemon en reserva
      [5] Probabilidad de sobrevivir 1 golpe del oponente (1 - P_KO)

    Total: 5 x 6 = 30 dims
    """
    result = np.zeros(30, dtype=np.float32)

    if opp_active is None:
        return result

    # Tipo principal del oponente (primer tipo no None)
    opp_primary_type = next((t for t in opp_active.types if t is not None), None)

    opp_spe = _real_speed(opp_active, battle)

    for i in range(5):
        offset = i * 6
        if i >= len(reserve_pokemon):
            continue
        poke = reserve_pokemon[i]
        if poke is None or poke.fainted:
            continue

        # [0] Mejor daño ofensivo vs oponente
        result[offset + 0] = float(_best_damage_vs(poke, opp_active, battle=battle, is_own_attack=True))

        # [1] Resiste tipo principal, [2] Inmune
        if opp_primary_type is not None:
            resist_mult = 1.0
            for def_type in poke.types:
                if def_type is not None and opp_primary_type in TYPE_CHART:
                    resist_mult *= TYPE_CHART[opp_primary_type].get(def_type, 1.0)
            if resist_mult == 0.0:
                result[offset + 2] = 1.0
                result[offset + 1] = 1.0
            elif resist_mult <= 0.5:
                result[offset + 1] = 1.0

        # [3] Ventaja de velocidad
        poke_spe = _real_speed(poke, battle)
        result[offset + 3] = 1.0 if poke_spe > opp_spe else 0.0

        # [4] HP fraction
        result[offset + 4] = float(poke.current_hp_fraction)

        # [5] Probabilidad de sobrevivir (1 - P_KO del oponente sobre este poke)
        opp_ko_prob = _ko_probability(opp_active, poke, battle=battle, is_own_attack=False)
        result[offset + 5] = float(1.0 - opp_ko_prob)

    return result


# ===========================================================================
# Analisis de combate
# ===========================================================================

def _encode_combat_analysis(own: Pokemon, opp: Pokemon, battle) -> np.ndarray:
    """
    36 dims:
      Por movimiento (4 x 8 = 32):
        [0-5] Efectividad one-hot (x0/x0.25/x0.5/x1/x2/x4)
        [6]   Daño estimado con formula Gen 6+ (0-1)
        [7]   Es movimiento de estado (base_power == 0)
      Global (4):
        [0] Somos mas rapidos (con Scarf y habilidades)
        [1] Probabilidad de que el oponente nos mate (P_KO) [0-1]
        [2] Probabilidad de que nosotros matemos al oponente (P_KO) [0-1]
        [3] Ventaja de tipo general (promedio effs / 4.0)
    """
    parts = []
    moves = list(own.moves.values())
    best_damage = 0.0

    for i in range(N_MOVES):
        move = moves[i] if i < len(moves) else None
        move_vec = np.zeros(8, dtype=np.float32)

        if move is not None and opp is not None:
            eff = _type_effectiveness(move.type, opp)

            # One-hot efectividad
            bucket_idx = 3
            for j, bucket in enumerate(EFFECTIVENESS_BUCKETS):
                if abs(eff - bucket) < 0.01:
                    bucket_idx = j
                    break
            move_vec[bucket_idx] = 1.0

            dmg = _estimate_damage(move, own, opp, battle=battle, is_own_attack=True)
            move_vec[6] = dmg
            best_damage = max(best_damage, dmg)
            move_vec[7] = 1.0 if move.base_power == 0 else 0.0

        parts.append(move_vec)

    global_vec = np.zeros(4, dtype=np.float32)
    if opp is not None:
        own_spe = _real_speed(own, battle)
        opp_spe = _real_speed(opp, battle)
        global_vec[0] = 1.0 if own_spe > opp_spe else 0.0

        # Probabilidades de KO con rango completo 0.85-1.0
        global_vec[1] = _ko_probability(opp, own, battle=battle, is_own_attack=False)
        global_vec[2] = _ko_probability(own, opp, battle=battle, is_own_attack=True)

        # Ventaja de tipo general
        effs = [_type_effectiveness(m.type, opp) for m in moves if m and m.base_power > 0]
        avg_eff = float(np.mean(effs)) if effs else 1.0
        global_vec[3] = min(avg_eff / 4.0, 1.0)

    parts.append(global_vec)
    return np.concatenate(parts)


# ===========================================================================
# Codificadores de pokemon
# ===========================================================================

def _encode_types(pokemon: Pokemon) -> np.ndarray:
    vec = np.zeros(len(TYPES), dtype=np.float32)
    for t in pokemon.types:
        if t is not None and t in TYPES:
            vec[TYPES.index(t)] = 1.0
    return vec


def _encode_status(pokemon: Pokemon) -> np.ndarray:
    vec = np.zeros(len(STATUSES) + 1, dtype=np.float32)
    if pokemon.status is None:
        vec[-1] = 1.0
    elif pokemon.status in STATUSES:
        vec[STATUSES.index(pokemon.status)] = 1.0
    return vec


def _encode_boosts(pokemon: Pokemon) -> np.ndarray:
    vec = np.zeros(len(BOOST_STATS), dtype=np.float32)
    for i, stat in enumerate(BOOST_STATS):
        vec[i] = pokemon.boosts.get(stat, 0) / 6.0
    return vec


def _encode_move(move: "Move | None", pokemon: Pokemon) -> np.ndarray:
    """tipo(18) + categoria(3) + potencia(1) + precision(1) + PP(1) + STAB(1) = 25"""
    if move is None:
        return np.zeros(25, dtype=np.float32)
    vec = np.zeros(25, dtype=np.float32)
    if move.type in TYPES:
        vec[TYPES.index(move.type)] = 1.0
    cat_map = {"physical": 0, "special": 1, "status": 2}
    vec[18 + cat_map.get(move.category.name.lower(), 2)] = 1.0
    vec[21] = min((move.base_power or 0) / 250.0, 1.0)
    acc = move.accuracy
    vec[22] = 1.0 if acc is True else (acc / 100.0 if acc else 0.0)
    if move.current_pp is not None and move.max_pp:
        vec[23] = move.current_pp / move.max_pp
    else:
        vec[23] = 1.0
    vec[24] = 1.0 if move.type in (pokemon.types or []) else 0.0
    return vec


def _encode_active_pokemon(pokemon: Pokemon, is_own: bool) -> np.ndarray:
    """HP(1) + tipos(18) + estado(7) + boosts(7) + stats(5) + moves(100) = 138"""
    parts = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _encode_types(pokemon),
        _encode_status(pokemon),
        _encode_boosts(pokemon),
    ]
    if is_own:
        stats = pokemon.base_stats
        parts.append(np.array([
            stats.get("atk", 0) / 255.0, stats.get("def", 0) / 255.0,
            stats.get("spa", 0) / 255.0, stats.get("spd", 0) / 255.0,
            stats.get("spe", 0) / 255.0,
        ], dtype=np.float32))
        moves = list(pokemon.moves.values())
        for i in range(N_MOVES):
            parts.append(_encode_move(moves[i] if i < len(moves) else None, pokemon))
    else:
        parts.append(np.zeros(5, dtype=np.float32))
        parts.append(np.zeros(N_MOVES * 25, dtype=np.float32))
    return np.concatenate(parts)


def _encode_reserve_pokemon(pokemon: "Pokemon | None") -> np.ndarray:
    """HP(1) + tipos(18) + disponible(1) = 20"""
    if pokemon is None:
        return np.zeros(20, dtype=np.float32)
    return np.concatenate([
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _encode_types(pokemon),
        np.array([float(not pokemon.fainted)], dtype=np.float32),
    ])


def _encode_field(battle: AbstractBattle) -> np.ndarray:
    """
    Campo de batalla: 28 dims total.

    clima(9) + terreno(5) + pantallas_propias(3) + pantallas_rivales(3)
    + hazards_propios(4) + hazards_rivales(4) = 28

    Hazards propios/rivales (4 dims cada uno):
      [0] Stealth Rock    (0 o 1)
      [1] Spikes          (0, 1/3, 2/3, 1)    — 0-3 capas
      [2] Toxic Spikes    (0, 0.5, 1)          — 0-2 capas
      [3] Sticky Web      (0 o 1)

    Estos son datos perfectamente observables en la batalla, pero el encoder
    anterior no los incluia. Son clave para:
      - Saber si cambiar es arriesgado (SR hace 25% a Charizard)
      - Valorar correctamente el switch analysis
      - Saber si el rival tiene presion de hazards para forzar switches
    """
    parts = []

    # Clima (9 = 8 tipos + sin clima)
    weather_vec = np.zeros(len(WEATHERS) + 1, dtype=np.float32)
    if battle.weather:
        wk = list(battle.weather.keys())[0]
        if wk in WEATHERS:
            weather_vec[WEATHERS.index(wk)] = 1.0
        else:
            weather_vec[-1] = 1.0
    else:
        weather_vec[-1] = 1.0
    parts.append(weather_vec)

    # Terreno (5 = 4 tipos + sin terreno)
    field_vec = np.zeros(len(FIELDS) + 1, dtype=np.float32)
    for f in battle.fields:
        if f in FIELDS:
            field_vec[FIELDS.index(f)] = 1.0
            break
    else:
        field_vec[-1] = 1.0
    parts.append(field_vec)

    # Pantallas propias (3)
    own_sides = battle.side_conditions
    parts.append(np.array([
        float(SideCondition.REFLECT      in own_sides),
        float(SideCondition.LIGHT_SCREEN in own_sides),
        float(SideCondition.AURORA_VEIL  in own_sides),
    ], dtype=np.float32))

    # Pantallas rivales (3)
    opp_sides = battle.opponent_side_conditions
    parts.append(np.array([
        float(SideCondition.REFLECT      in opp_sides),
        float(SideCondition.LIGHT_SCREEN in opp_sides),
        float(SideCondition.AURORA_VEIL  in opp_sides),
    ], dtype=np.float32))

    # Hazards propios (4) — los que afectan a NUESTROS switches
    sr_own    = float(SideCondition.STEALTH_ROCK in own_sides)
    spk_own   = own_sides.get(SideCondition.SPIKES, 0) / 3.0          # 0-3 capas → 0-1
    tspk_own  = own_sides.get(SideCondition.TOXIC_SPIKES, 0) / 2.0    # 0-2 capas → 0-1
    web_own   = float(SideCondition.STICKY_WEB in own_sides)
    parts.append(np.array([sr_own, spk_own, tspk_own, web_own], dtype=np.float32))

    # Hazards rivales (4) — los que afectan a los switches del RIVAL
    sr_opp    = float(SideCondition.STEALTH_ROCK in opp_sides)
    spk_opp   = opp_sides.get(SideCondition.SPIKES, 0) / 3.0
    tspk_opp  = opp_sides.get(SideCondition.TOXIC_SPIKES, 0) / 2.0
    web_opp   = float(SideCondition.STICKY_WEB in opp_sides)
    parts.append(np.array([sr_opp, spk_opp, tspk_opp, web_opp], dtype=np.float32))

    return np.concatenate(parts)


# ===========================================================================
# Punto de entrada principal
# ===========================================================================

def encode_battle(battle: AbstractBattle) -> np.ndarray:
    """
    Devuelve el vector de observacion completo (570 dims).

      1. Activo propio     (138): HP + tipos + estado + boosts + stats + moves
      2. Combate           ( 36): 4 moves x 8 + 4 globales (formula Gen 6+)
      3. Cambios           ( 30): 5 reservas x 6 dims
      4. Reserva propia    (100): 5 x 20
      5. Activo enemigo    (138)
      6. Reserva enemiga   (100)
      7. Campo             ( 28): clima + terreno + pantallas + hazards
      TOTAL: 570
    """
    own_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    own_team   = [p for p in battle.team.values()          if not p.active]
    opp_team   = [p for p in battle.opponent_team.values() if not p.active]

    parts = []

    # 1. Activo propio (138)
    parts.append(
        _encode_active_pokemon(own_active, is_own=True)
        if own_active else np.zeros(_active_pokemon_size(), dtype=np.float32)
    )

    # 2. Analisis de combate (36)
    parts.append(
        _encode_combat_analysis(own_active, opp_active, battle)
        if own_active and opp_active else np.zeros(_combat_analysis_size(), dtype=np.float32)
    )

    # 3. Analisis de cambios (30)
    reserve_list = [own_team[i] if i < len(own_team) else None for i in range(5)]
    parts.append(
        _encode_switch_analysis(reserve_list, opp_active, battle=battle)
        if opp_active else np.zeros(_switch_analysis_size(), dtype=np.float32)
    )

    # 4. Reserva propia (100)
    for i in range(5):
        parts.append(_encode_reserve_pokemon(own_team[i] if i < len(own_team) else None))

    # 5. Activo enemigo (138)
    parts.append(
        _encode_active_pokemon(opp_active, is_own=False)
        if opp_active else np.zeros(_active_pokemon_size(), dtype=np.float32)
    )

    # 6. Reserva enemiga (100)
    for i in range(5):
        parts.append(_encode_reserve_pokemon(opp_team[i] if i < len(opp_team) else None))

    # 7. Campo (28)
    parts.append(_encode_field(battle))

    return np.concatenate(parts).astype(np.float32)


# ===========================================================================
# Helpers de tamaño
# ===========================================================================

def _active_pokemon_size() -> int:
    return 1 + 18 + 7 + 7 + 5 + (N_MOVES * 25)   # = 138


def _combat_analysis_size() -> int:
    return N_MOVES * 8 + 4   # = 36


def _switch_analysis_size() -> int:
    return 5 * 6   # = 30


def get_observation_size() -> int:
    """Devuelve el tamaño total del vector de observacion (570)."""
    active  = _active_pokemon_size()   # 138
    combat  = _combat_analysis_size()  #  36
    switch  = _switch_analysis_size()  #  30
    reserve = 20
    field   = 9 + 5 + 3 + 3 + 4 + 4  #  28  (+ hazards propios + hazards rivales)
    return (
        active +       # propio activo    138
        combat +       # combate           36
        switch +       # cambios           30
        5 * reserve +  # reserva propia   100
        active +       # enemigo activo   138
        5 * reserve +  # reserva enemiga  100
        field          # campo             28
    )   # = 570


OBS_SIZE = get_observation_size()
