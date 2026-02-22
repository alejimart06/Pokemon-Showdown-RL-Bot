"""
items_and_abilities.py
======================
Referencia completa de items y habilidades de Pokemon, con su efecto
sobre el calculo de daño, velocidad y otros mecanismos del bot.

Importado por encoder.py para centralizar toda la logica de items/habilidades.

ESTRUCTURA:
  ATTACK_ITEMS   — items que modifican el daño del ATACANTE
  DEFENSE_ITEMS  — items que modifican el daño recibido por el DEFENSOR
  SPEED_ITEMS    — items que modifican la velocidad
  TYPE_BOOST_ITEMS — items que potencian un tipo concreto (x1.2)
  ATTACK_ABILITIES — habilidades del atacante
  DEFENSE_ABILITIES — habilidades del defensor (inmunidades y reducciones)
  SPEED_ABILITIES  — habilidades que afectan la velocidad

CONVENCION DE VALORES:
  - mult > 1.0 => aumenta el daño
  - mult < 1.0 => reduce el daño (en items de defensa significa divisor)
  - "immune" como string => inmune al tipo (daño = 0)
  - Los items de tipo (x1.2) se buscan en TYPE_BOOST_ITEMS adicionalmente

FUENTES:
  Bulbapedia: https://bulbapedia.bulbagarden.net/wiki/Item
  Smogon:     https://www.smogon.com/dex/
  Gen 9 (Scarlet/Violet), Gen 8 (S/S), Gen 7 (S/M), Gen 6 (X/Y)
"""

from poke_env.battle import PokemonType


# ===========================================================================
# CONJUNTOS DE MOVIMIENTOS (usados por habilidades y items)
# ===========================================================================

# Movimientos de puñetazo (Iron Fist, Punch Glove)
PUNCH_MOVES: set[str] = {
    "bulletpunch", "cometpunch", "dizzypunch", "drainpunch", "dynamicpunch",
    "firepunch", "focuspunch", "hammerarm", "icehammer", "icepunch",
    "machpunch", "megapunch", "meteormash", "poisonjab", "poweruppunch",
    "shadowpunch", "skyuppercut", "superpower", "thunderpunch",
    "zippyzap", "firstimpression",
}

# Movimientos de retroceso (Reckless)
RECOIL_MOVES: set[str] = {
    "bravebird", "doubleedge", "flareblitz", "headsmash", "highjumpkick",
    "jumpkick", "submission", "takedown", "volttackle", "wildcharge",
    "woodhammer", "headcharge",
}

# Movimientos de mordisco (Strong Jaw)
BITE_MOVES: set[str] = {
    "bite", "crunch", "firefang", "hyperfang", "icefang", "poisonfang",
    "psychicfangs", "thunderfang",
}

# Movimientos de pulso/aura (Mega Launcher)
PULSE_MOVES: set[str] = {
    "aurasphere", "darkpulse", "dragonpulse", "healpulse", "hokeypokey",
    "lusterpurge", "mindblown", "moonblast", "originpulse", "terrainpulse",
    "waterpulse",
}

# Movimientos de sonido (Punk Rock, Soundproof)
SOUND_MOVES: set[str] = {
    "boomburst", "bugbuzz", "chatter", "clangingscales", "clangoroussoul",
    "disarmingvoice", "echoedvoice", "grasswhistle", "growl", "hypervoice",
    "metalsound", "nobleroar", "overdrive", "partingshot", "perishsong",
    "relicsong", "roar", "round", "screech", "sing", "snarl", "snore",
    "sparklingsong", "supersonic", "uproar", "healbell",
}

# Movimientos de contacto fisico (Tough Claws, Rough Skin, etc.)
CONTACT_MOVES: set[str] = {
    "aquajet", "aquastep", "bayonetcharge", "bodypress", "bodyslam",
    "bravebird", "bulldoze", "closecombat", "crunch", "doubleedge",
    "drainpunch", "dragonrush", "dragonclaw", "dragonwhip", "extremespeed",
    "facade", "falseswipe", "flareblitz", "flyingpress", "focuspunch",
    "geargrind", "headsmash", "highjumpkick", "iciclecrash", "ironhead",
    "leafblade", "lowkick", "lowsweep", "lunge", "nightslash", "outrage",
    "phantomforce", "playrough", "poisonjab", "psychocut", "psychicfangs",
    "rapidspin", "shadowclaw", "shadowsneak", "slash", "stoneedge",
    "submission", "superpower", "takedown", "thunderpunch",
    "uturn", "woodhammer", "xscissor", "zenheadbutt",
} | PUNCH_MOVES | BITE_MOVES

# Movimientos de bola/bomba (Bulletproof)
BALL_BOMB_MOVES: set[str] = {
    "aurasphere", "barrage", "eggbomb", "electroball", "energyball",
    "focusblast", "gunkshot", "gyroball", "iceball", "leafstorm",
    "magnetbomb", "mistball", "mudbomb", "octazooka", "payday",
    "pollenpuff", "rockblast", "rockwrecker", "seedbomb", "seedflare",
    "shadowball", "sludgebomb", "syrupbomb", "weatherball",
}

# ===========================================================================
# HELPERS
# ===========================================================================

def clean(s) -> str | None:
    """Normaliza nombre de item/habilidad para comparacion uniforme."""
    if s is None:
        return None
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


# ===========================================================================
# ITEMS DE ATAQUE (modifican el daño del atacante)
# ===========================================================================
# Formato: "nombre_limpio": {"mult": float, "category": "physical"|"special"|"both"|None}
# category=None => aplica a todos los movimientos de daño
# mult es el multiplicador SOBRE EL DAÑO FINAL (no sobre el stat)

ATTACK_ITEMS: dict[str, dict] = {
    # ---- Potenciadores globales ----
    "lifeorb":       {"mult": 1.3,  "category": None},       # x1.3 daño (recoil 10% HP)
    "muscleband":    {"mult": 1.1,  "category": "physical"},  # x1.1 fisico
    "wiseglasses":   {"mult": 1.1,  "category": "special"},   # x1.1 especial
    "expertbelt":    {"mult": 1.2,  "category": None, "supereffective_only": True},  # x1.2 SE

    # ---- Choice items: x1.5 al stat de ataque relevante ----
    # (implementados como multiplicador de daño equivalente)
    "choiceband":    {"mult": 1.5,  "category": "physical"},
    "choicespecs":   {"mult": 1.5,  "category": "special"},
    # Choice Scarf solo afecta velocidad, no daño
    "choicescarf":   {"mult": 1.0,  "category": None},

    # ---- Orbes especificos de legendarios ----
    "adamantorb":    {"mult": 1.2,  "category": None, "user_species": {"dialga"},
                      "types": {PokemonType.DRAGON, PokemonType.STEEL}},
    "lustrousorb":   {"mult": 1.2,  "category": None, "user_species": {"palkia"},
                      "types": {PokemonType.DRAGON, PokemonType.WATER}},
    "griseousorb":   {"mult": 1.2,  "category": None, "user_species": {"giratina"},
                      "types": {PokemonType.DRAGON, PokemonType.GHOST}},
    "souldew":       {"mult": 1.2,  "category": None, "user_species": {"latios", "latias"},
                      "types": {PokemonType.DRAGON, PokemonType.PSYCHIC}},

    # ---- Punch Glove (Gen 9): x1.1 puñetazos, elimina contacto ----
    "punchingglove": {"mult": 1.1,  "category": "physical", "punch_only": True},

    # ---- Throat Spray (Gen 8): x1.5 SpA tras movimiento de sonido ----
    # (efecto reactivo, dificil de modelar sin estado; incluido como referencia)
    "throatspray":   {"mult": 1.0,  "category": "special", "note": "reactive"},

    # ---- Metronome: aumenta segun usos consecutivos (simplificado) ----
    "metronome":     {"mult": 1.0,  "category": None, "note": "stackable"},

    # ---- Terrain seeds: no afectan daño directamente ----
    # ---- Power items (para cria, no combate) ----
}


# ===========================================================================
# ITEMS DE DEFENSA (modifican el daño recibido por el defensor)
# Formato: "nombre": {"divisor": float, "category": "physical"|"special"|"both"|None, "type": PokemonType|None}
# divisor > 1.0 => reduce el daño (daño_final / divisor)
# ===========================================================================

DEFENSE_ITEMS: dict[str, dict] = {
    # ---- Potenciadores de defensa (reducen daño recibido) ----
    "eviolite":      {"divisor": 1.5,  "category": None},       # x1.5 Def y SpD si no es final
    "assaultvest":   {"divisor": 1.5,  "category": "special"},  # x1.5 SpD (no puede usar status)

    # ---- Berries de reduccion de daño ----
    # (se consumen, efectos reactivos; incluidas como referencia)
    "occaberry":     {"divisor": 2.0,  "category": "physical", "type": PokemonType.FIRE,
                      "note": "consumed_on_hit"},
    "passhoberry":   {"divisor": 2.0,  "category": "special",  "type": PokemonType.WATER,
                      "note": "consumed_on_hit"},
    "wacanberry":    {"divisor": 2.0,  "category": "special",  "type": PokemonType.ELECTRIC,
                      "note": "consumed_on_hit"},
    "rindoberry":    {"divisor": 2.0,  "category": "special",  "type": PokemonType.GRASS,
                      "note": "consumed_on_hit"},
    "yacheberry":    {"divisor": 2.0,  "category": "special",  "type": PokemonType.ICE,
                      "note": "consumed_on_hit"},
    "chopleberry":   {"divisor": 2.0,  "category": "physical", "type": PokemonType.FIGHTING,
                      "note": "consumed_on_hit"},
    "kebiaberry":    {"divisor": 2.0,  "category": "physical", "type": PokemonType.POISON,
                      "note": "consumed_on_hit"},
    "shucaberry":    {"divisor": 2.0,  "category": "physical", "type": PokemonType.GROUND,
                      "note": "consumed_on_hit"},
    "cobaberry":     {"divisor": 2.0,  "category": "physical", "type": PokemonType.FLYING,
                      "note": "consumed_on_hit"},
    "payapaberry":   {"divisor": 2.0,  "category": "special",  "type": PokemonType.PSYCHIC,
                      "note": "consumed_on_hit"},
    "tangaberry":    {"divisor": 2.0,  "category": "physical", "type": PokemonType.BUG,
                      "note": "consumed_on_hit"},
    "chartiberry":   {"divisor": 2.0,  "category": "special",  "type": PokemonType.ROCK,
                      "note": "consumed_on_hit"},
    "kasibberry":    {"divisor": 2.0,  "category": "special",  "type": PokemonType.GHOST,
                      "note": "consumed_on_hit"},
    "habanberry":    {"divisor": 2.0,  "category": "special",  "type": PokemonType.DRAGON,
                      "note": "consumed_on_hit"},
    "colburberry":   {"divisor": 2.0,  "category": "special",  "type": PokemonType.DARK,
                      "note": "consumed_on_hit"},
    "babiriberry":   {"divisor": 2.0,  "category": "physical", "type": PokemonType.STEEL,
                      "note": "consumed_on_hit"},
    "chilanberry":   {"divisor": 2.0,  "category": "physical", "type": PokemonType.NORMAL,
                      "note": "consumed_on_hit"},
    "roseliberry":   {"divisor": 2.0,  "category": "special",  "type": PokemonType.FAIRY,
                      "note": "consumed_on_hit"},

    # ---- Rocky Helmet: 1/6 HP al atacante por contacto, no afecta daño recibido ----
    "rockyhelmet":   {"divisor": 1.0,  "category": None, "note": "recoil_to_attacker"},
}


# ===========================================================================
# ITEMS DE VELOCIDAD
# Formato: "nombre": {"mult": float}
# ===========================================================================

SPEED_ITEMS: dict[str, dict] = {
    "choicescarf":   {"mult": 1.5},   # x1.5 velocidad, bloquea a 1 movimiento
    "ironball":      {"mult": 0.5},   # x0.5 velocidad, anula Levitate/vuelo
    "machobrace":    {"mult": 0.5},   # x0.5 velocidad (entrenamiento, raro en combate)
    "powerweight":   {"mult": 0.5},
    "powerbracer":   {"mult": 0.5},
    "powerbelt":     {"mult": 0.5},
    "powerlens":     {"mult": 0.5},
    "powerband":     {"mult": 0.5},
    "poweranklet":   {"mult": 0.5},
    "quickpowder":   {"mult": 2.0,  "user_species": {"ditto"}},  # solo Ditto
    "fullincense":   {"mult": 0.5,  "note": "moves_last"},       # mueve al final
    "laggingitail":  {"mult": 0.5,  "note": "moves_last"},
    "salacberry":    {"mult": 1.5,  "note": "reactive_low_hp"},  # activacion al 25% HP
    "etherberry":    {"mult": 1.0,  "note": "reactive"},         # no afecta velocidad directamente
}


# ===========================================================================
# ITEMS DE TIPO (x1.2 al movimiento del tipo correspondiente)
# Incluye todas las placas de Arceus + objetos especificos
# Formato: "nombre_limpio": PokemonType
# ===========================================================================

TYPE_BOOST_ITEMS: dict[str, PokemonType] = {
    # --- Normal ---
    "silkscarf":      PokemonType.NORMAL,
    "normalplate":    PokemonType.NORMAL,   # Arceus-Normal

    # --- Fighting ---
    "blackbelt":      PokemonType.FIGHTING,
    "fistplate":      PokemonType.FIGHTING,  # Arceus-Fighting
    "blackgloveitem": PokemonType.FIGHTING,  # nombre interno a veces

    # --- Flying ---
    "sharpbeak":      PokemonType.FLYING,
    "skyplate":       PokemonType.FLYING,    # Arceus-Flying

    # --- Poison ---
    "poisonbarb":     PokemonType.POISON,
    "toxicplate":     PokemonType.POISON,    # Arceus-Poison

    # --- Ground ---
    "softsand":       PokemonType.GROUND,
    "earthplate":     PokemonType.GROUND,    # Arceus-Ground

    # --- Rock ---
    "hardstone":      PokemonType.ROCK,
    "stoneplate":     PokemonType.ROCK,      # Arceus-Rock

    # --- Bug ---
    "silverpowder":   PokemonType.BUG,
    "insectplate":    PokemonType.BUG,       # Arceus-Bug

    # --- Ghost ---
    "spelltag":       PokemonType.GHOST,
    "spookyplate":    PokemonType.GHOST,     # Arceus-Ghost

    # --- Steel ---
    "metalcoat":      PokemonType.STEEL,
    "ironplate":      PokemonType.STEEL,     # Arceus-Steel

    # --- Fire ---
    "charcoal":       PokemonType.FIRE,
    "flameplate":     PokemonType.FIRE,      # Arceus-Fire

    # --- Water ---
    "mysticwater":    PokemonType.WATER,
    "splashplate":    PokemonType.WATER,     # Arceus-Water
    "seaincense":     PokemonType.WATER,
    "waveincense":    PokemonType.WATER,

    # --- Grass ---
    "miracleseed":    PokemonType.GRASS,
    "meadowplate":    PokemonType.GRASS,     # Arceus-Grass
    "roseincense":    PokemonType.GRASS,

    # --- Electric ---
    "magnet":         PokemonType.ELECTRIC,
    "zapplate":       PokemonType.ELECTRIC,  # Arceus-Electric

    # --- Psychic ---
    "twistedspoon":   PokemonType.PSYCHIC,
    "mindplate":      PokemonType.PSYCHIC,   # Arceus-Psychic
    "oddincense":     PokemonType.PSYCHIC,

    # --- Ice ---
    "nevermeltice":   PokemonType.ICE,
    "icicleplate":    PokemonType.ICE,       # Arceus-Ice

    # --- Dragon ---
    "dragonfang":     PokemonType.DRAGON,
    "dracoplate":     PokemonType.DRAGON,    # Arceus-Dragon

    # --- Dark ---
    "blackglasses":   PokemonType.DARK,
    "dreadplate":     PokemonType.DARK,      # Arceus-Dark

    # --- Fairy ---
    "fairyfeather":   PokemonType.FAIRY,
    "pixieplate":     PokemonType.FAIRY,     # Arceus-Fairy

    # --- Orbes de legendarios (solo para esa especie) ---
    "adamantorb":     PokemonType.DRAGON,    # Dialga: Dragon y Steel
    "lustrousorb":    PokemonType.DRAGON,    # Palkia: Dragon y Water
    "griseousorb":    PokemonType.DRAGON,    # Giratina: Dragon y Ghost
    "souldew":        PokemonType.DRAGON,    # Latios/Latias: Dragon y Psychic
}

# Tipos secundarios de items de legendarios (para doble bono)
LEGENDARY_ORB_SECONDARY: dict[str, PokemonType] = {
    "adamantorb": PokemonType.STEEL,
    "lustrousorb": PokemonType.WATER,
    "griseousorb": PokemonType.GHOST,
    "souldew":     PokemonType.PSYCHIC,
}

# Especies a las que aplica cada item legendario
LEGENDARY_ORB_SPECIES: dict[str, set] = {
    "adamantorb": {"dialga"},
    "lustrousorb": {"palkia"},
    "griseousorb": {"giratina", "giratinaorigin"},
    "souldew":     {"latios", "latias", "latiosm", "latiosm"},
}


# ===========================================================================
# HABILIDADES DEL ATACANTE
# Formato: { "habilidad": {...datos...} }
# ===========================================================================
# Campos posibles:
#   mult      : multiplicador de daño (float)
#   category  : "physical"|"special"|None (None = ambas)
#   type      : PokemonType o set de tipos al que aplica
#   condition : descripcion de condicion de activacion
#   stab_mult : si no es None, reemplaza el multiplicador de STAB
#   move_set  : nombre del set de movimientos en encoder (PUNCH_MOVES, etc.)
#   always_stab: True si siempre tiene STAB (Protean/Libero)

ATTACK_ABILITIES: dict[str, dict] = {
    # ==== Modificadores de STAB ====
    "adaptability":   {"stab_mult": 2.0,  "condition": "always"},
    "protean":        {"always_stab": True, "stab_mult": 1.5},   # tipo cambia al del movimiento
    "libero":         {"always_stab": True, "stab_mult": 1.5},   # identico a Protean

    # ==== Multiplicadores de daño de tipo ====
    "steelworker":    {"mult": 1.5,  "type": PokemonType.STEEL},
    "transistor":     {"mult": 1.5,  "type": PokemonType.ELECTRIC},
    "dragonsmaw":     {"mult": 1.5,  "type": PokemonType.DRAGON},
    "rockypayload":   {"mult": 1.5,  "type": PokemonType.ROCK},
    "hadronengine":   {"mult": 1.333, "type": PokemonType.ELECTRIC,
                       "condition": "electric_terrain"},         # Gen 9: x4/3 en Electric Terrain
    "orichalcumpulse":{"mult": 1.333, "type": PokemonType.FIRE,
                       "condition": "sunny"},                    # Gen 9: x4/3 en sol

    # ==== Potenciadores por categoria ====
    "hustle":         {"mult": 1.5,  "category": "physical", "accuracy_penalty": 0.8},
    "gorilltatactics":{"mult": 1.5,  "category": "physical", "condition": "locks_to_one_move"},

    # ==== Potenciadores condicionales (stat bajo) ====
    "blaze":          {"mult": 1.5,  "type": PokemonType.FIRE,     "condition": "hp_below_third"},
    "torrent":        {"mult": 1.5,  "type": PokemonType.WATER,    "condition": "hp_below_third"},
    "overgrow":       {"mult": 1.5,  "type": PokemonType.GRASS,    "condition": "hp_below_third"},
    "swarm":          {"mult": 1.5,  "type": PokemonType.BUG,      "condition": "hp_below_third"},

    # ==== Potenciadores por estado propio ====
    "guts":           {"mult": 1.5,  "category": "physical",       "condition": "status", "negates_burn": True},
    "marvelscale":    {"mult": 1.0,  "condition": "status",         "note": "defensa, no ataque"},
    "quickfeet":      {"mult": 1.0,  "condition": "status",         "note": "velocidad, no ataque"},
    "flareBoost":     {"mult": 1.5,  "category": "special",         "condition": "burned"},
    "toxicboost":     {"mult": 1.5,  "category": "physical",        "condition": "poisoned"},

    # ==== Potenciadores por tipo de movimiento (set especifico) ====
    "technician":     {"mult": 1.5,  "condition": "base_power_60_or_less"},
    "sheerforce":     {"mult": 1.3,  "condition": "move_has_secondary"},
    "reckless":       {"mult": 1.2,  "move_set": "RECOIL_MOVES"},
    "ironfist":       {"mult": 1.2,  "move_set": "PUNCH_MOVES"},
    "strongjaw":      {"mult": 1.5,  "move_set": "BITE_MOVES"},
    "megalauncher":   {"mult": 1.5,  "move_set": "PULSE_MOVES"},
    "toughclaws":     {"mult": 1.3,  "move_set": "CONTACT_MOVES"},
    "punkrock":       {"mult": 1.3,  "move_set": "SOUND_MOVES"},
    "sandforce":      {"mult": 1.3,  "type": {PokemonType.ROCK, PokemonType.STEEL, PokemonType.GROUND},
                       "condition": "sandstorm", "category": "physical"},
    "analytic":       {"mult": 1.3,  "condition": "moves_last"},   # x1.3 si el rival actua primero
    "tintedlens":     {"mult": 2.0,  "condition": "not_very_effective"},  # NVE -> x1 efectivo
    "aerilate":       {"mult": 1.2,  "type_change": (PokemonType.NORMAL, PokemonType.FLYING)},
    "pixilate":       {"mult": 1.2,  "type_change": (PokemonType.NORMAL, PokemonType.FAIRY)},
    "refrigerate":    {"mult": 1.2,  "type_change": (PokemonType.NORMAL, PokemonType.ICE)},
    "galvanize":      {"mult": 1.2,  "type_change": (PokemonType.NORMAL, PokemonType.ELECTRIC)},
    "liquidvoice":    {"mult": 1.0,  "type_change": ("sound", PokemonType.WATER)},  # sonido -> Water
    "normalize":      {"mult": 1.0,  "type_change": ("all", PokemonType.NORMAL)},   # todo -> Normal
    "electricsurge":  {"mult": 1.0,  "note": "sets_electric_terrain"},
    "psychicsurge":   {"mult": 1.0,  "note": "sets_psychic_terrain"},

    # ==== Potenciadores especiales ====
    "neuroforce":     {"mult": 1.25, "condition": "supereffective"},      # x1.25 SE
    "sniper":         {"mult": 1.5,  "condition": "critical_hit"},        # criticos x1.5 extra
    "supremeoverlord":{"mult": 1.1,  "condition": "per_fainted_ally",     # +10% por caido
                       "note": "max_5_stacks"},
    "powerspot":      {"mult": 1.3,  "note": "doubles_only"},
    "steelyspirit":   {"mult": 1.5,  "type": PokemonType.STEEL, "note": "ally_boost_doubles"},

    # ==== Clima especifico ====
    "solarpower":     {"mult": 1.5,  "category": "special",   "condition": "sunny",
                       "hp_cost": 0.125},
    "sandstorm":      {"mult": 1.0,  "note": "sets_sandstorm"},
    "snowwarning":    {"mult": 1.0,  "note": "sets_snow_gen9"},

    # ==== Otros ====
    "deathmask":      {"mult": 1.3,  "condition": "opponent_fainted_last_turn"},  # hypothetico
    "stench":         {"mult": 1.0,  "note": "flinch_10pct"},  # no afecta daño
}


# ===========================================================================
# HABILIDADES DEL DEFENSOR
# Formato: { "habilidad": {...datos...} }
# Campos:
#   immune     : PokemonType o set de tipos => inmune a ese tipo
#   divisor    : float > 1 => divide el daño recibido
#   mult       : float < 1 => multiplica el daño (aumenta si < 1 significa mas daño)
#   category   : "physical"|"special"|None
#   type       : PokemonType al que aplica la reduccion
#   condition  : condicion de activacion
# ===========================================================================

DEFENSE_ABILITIES: dict[str, dict] = {
    # ==== Inmunidades de tipo ====
    "levitate":       {"immune": PokemonType.GROUND,    "note": "anulada_por_gravity_ringout_ironball"},
    "flashfire":      {"immune": PokemonType.FIRE,      "note": "boost_fire_tras_absorber"},
    "waterabsorb":    {"immune": PokemonType.WATER,     "note": "cura_25pct_hp"},
    "dryskin":        {"immune": PokemonType.WATER,     "mult": 1.25, "type": PokemonType.FIRE,
                       "note": "recibe_mas_fuego_se_cura_con_agua_y_lluvia"},
    "stormdrain":     {"immune": PokemonType.WATER,     "note": "boost_spa_tras_absorber"},
    "voltabsorb":     {"immune": PokemonType.ELECTRIC,  "note": "cura_25pct_hp"},
    "motordrive":     {"immune": PokemonType.ELECTRIC,  "note": "boost_spe_tras_absorber"},
    "lightningrod":   {"immune": PokemonType.ELECTRIC,  "note": "boost_spa_en_doubles"},
    "sapsipper":      {"immune": PokemonType.GRASS,     "note": "boost_atk_tras_absorber"},
    "eartheater":     {"immune": PokemonType.GROUND,    "note": "gen9_cura_25pct_hp"},
    "windrider":      {"immune": PokemonType.FLYING,    "note": "gen9_boost_atk"},    # Tailwind e inmunidad Wind
    "wellbakedbody":  {"immune": PokemonType.FIRE,      "note": "gen9_boost_def"},
    "steamengine":    {"immune": PokemonType.WATER,     "note": "boost_spe_tras_absorber"},  # tambien Fire
    "bulletproof":    {"immune_move_set": "BALL_BOMB_MOVES",               "note": "inmune_a_balas_bombas"},
    "soundproof":     {"immune_move_set": "SOUND_MOVES",                   "note": "inmune_a_sonido"},
    "telepathy":      {"immune": "ally_moves",                              "note": "doubles_only"},
    "suctioncups":    {"immune": "phazing",                                 "note": "no_es_arrastrado"},
    "overcoat":       {"immune": "weather_damage",                          "note": "y_movimientos_de_polvo"},
    "magicguard":     {"immune": "indirect_damage",                         "note": "solo_daño_directo"},
    "wonderguard":    {"condition": "immune_unless_supereffective"},

    # ==== Reducciones de daño por tipo/categoria ====
    "thickfat":       {"divisor": 2.0,  "type": {PokemonType.FIRE, PokemonType.ICE}},
    "heatproof":      {"divisor": 2.0,  "type": PokemonType.FIRE},
    "purifyingsalt":  {"divisor": 2.0,  "type": PokemonType.GHOST},
    "icescales":      {"divisor": 2.0,  "category": "special"},          # x0.5 daño especial
    "multiscale":     {"divisor": 2.0,  "condition": "full_hp"},         # x0.5 con HP lleno
    "shadowshield":   {"divisor": 2.0,  "condition": "full_hp"},         # igual que Multiscale
    "filter":         {"divisor": 1.333, "condition": "supereffective"},  # x0.75 SE (= /1.333)
    "solidrock":      {"divisor": 1.333, "condition": "supereffective"},
    "prismarmor":     {"divisor": 1.333, "condition": "supereffective"},
    "fluffy":         {"divisor": 2.0,  "category": "physical", "contact_only": True,
                       "extra": {"mult": 2.0, "type": PokemonType.FIRE}},  # Fire x2 pero contacto x0.5
    "punkrock":       {"divisor": 2.0,  "move_set": "SOUND_MOVES"},
    "fairyaura":      {"note": "mult_fairy_for_all",                        "note2": "doubles_mainly"},
    "darkaura":       {"note": "mult_dark_for_all"},
    "aurabreak":      {"note": "inverts_fairy_dark_aura"},

    # ==== Modificadores de defensa (aumentan stat defensivo efectivo) ====
    # Nota: Fur Coat y Marvel Scale son HABILIDADES, no items
    "furcoat":        {"divisor": 2.0,  "category": "physical"},         # x0.5 daño fisico recibido
    "marvelscale":    {"divisor": 1.5,  "category": "physical",          "condition": "status"},
    "grasspelt":      {"divisor": 1.5,  "category": "physical",          "condition": "grassy_terrain"},

    # ==== Inmunidades condicionales ====
    "dazzling":       {"immune": "priority_moves",                         "note": "prioridad_rival"},
    "queenlymajesty": {"immune": "priority_moves"},
    "armortail":      {"immune": "priority_moves",                         "note": "gen9"},
    "goodasgold":     {"immune": "status_moves",                           "note": "gen9"},
    "guarddog":       {"immune": "intimidate",                             "note": "gen9_boost_atk_instead"},

    # ==== Efectos post-dano (no reducen el daño directamente) ====
    "roughskin":      {"note": "1/8_hp_recoil_to_attacker_on_contact"},
    "ironbarbs":      {"note": "1/8_hp_recoil_to_attacker_on_contact"},
    "rockyhelmet":    {"note": "ITEM_no_habilidad"},  # marcado aqui para referencia
    "cottondown":     {"note": "reduce_spe_atacante_en_1"},
    "perishbody":     {"note": "ambos_faint_en_3_turnos_si_contacto"},
    "wanderingspirit":{"note": "intercambia_habilidad_con_atacante_en_contacto"},
    "gooey":          {"note": "reduce_spe_atacante_en_1_por_contacto"},
    "tanglinghair":   {"note": "reduce_spe_atacante_en_1_por_contacto"},
}


# ===========================================================================
# HABILIDADES DE VELOCIDAD
# Formato: { "habilidad": {"mult": float, "condition": str} }
# ===========================================================================

SPEED_ABILITIES: dict[str, dict] = {
    "swiftswim":     {"mult": 2.0,  "condition": "rain"},
    "chlorophyll":   {"mult": 2.0,  "condition": "sun"},
    "sandrush":      {"mult": 2.0,  "condition": "sandstorm"},
    "slushrush":     {"mult": 2.0,  "condition": "hail_or_snow"},
    "surgesurfer":   {"mult": 2.0,  "condition": "electric_terrain"},
    "speedboost":    {"mult": 1.0,  "condition": "per_turn",  "note": "+1 spe al final de cada turno"},
    "unburden":      {"mult": 2.0,  "condition": "item_lost"},
    "quickfeet":     {"mult": 1.5,  "condition": "status"},
    "slowstart":     {"mult": 0.5,  "condition": "first_5_turns"},
    "stall":         {"mult": 0.5,  "note": "mueve_al_final"},
    "truant":        {"mult": 1.0,  "note": "solo_actua_cada_2_turnos"},
    "prankster":     {"mult": 1.0,  "note": "prioridad_+1_status_moves"},
    "triage":        {"mult": 1.0,  "note": "prioridad_+3_curacion"},
    "galewings":     {"mult": 1.0,  "note": "prioridad_+1_flying_a_full_hp"},
    "queenofwings":  {"mult": 1.0,  "note": "prioridad_flying_en_viento"},
}


# ===========================================================================
# HABILIDADES QUE AFECTAN AL STAT DE ATAQUE DEL RIVAL (Intimidate, etc.)
# Estas se aplican al momento del switch, modificando los boosts
# ===========================================================================

STAT_DROP_ABILITIES: dict[str, dict] = {
    "intimidate":    {"stat": "atk",  "stages": -1, "target": "opponent"},
    "cottondown":    {"stat": "spe",  "stages": -1, "target": "opponent"},
    "icyscale":      {"stat": "spa",  "stages": -1, "target": "opponent",  "note": "gen9"},
    "electromorphosis": {"note": "carga_un_bono_de_electrico",             "note2": "no_stat_drop"},
}


# ===========================================================================
# MOVIMIENTOS DE BOLA/BOMBA (para Bulletproof)
# ===========================================================================

BALL_BOMB_MOVES: set[str] = {
    "acidspray", "aurasphere", "barrage", "beachbomb", "boulderheave",
    "cannonball", "darkpulse",  # No: Dark Pulse no es bola
    "eggbomb", "electroball", "energyball", "focusblast", "gunkshot",
    "gyroball", "iceballmove", "icyball", "magicroom", "magnetbomb",
    "mistball", "mudbomb", "octazooka", "particlestorm", "payday",
    "rockblast", "rockwrecker", "seedbomb", "seedflare", "shadowball",
    "smellingsalts", "sludgebomb", "solarbeam", "syrupbomb", "thunderball",
    "weatherball", "zapbomb", "zingzap",
    # Canonicamente en la lista de Bulletproof:
    "aurasphere", "barrage", "beachbomb", "dragonball", "eggbomb",
    "electroball", "energyball", "focusblast", "gunkshot", "gyroball",
    "iceball", "leafstorm", "magnetbomb", "mistball", "mudbomb",
    "octazooka", "payday", "rockblast", "rockwrecker", "seedbomb",
    "seedflare", "shadowball", "sludgebomb", "weatherball",
    "pollenpuff", "syrupbomb", "maliciousmoonsault",
}


# ===========================================================================
# FUNCIONES DE CONSULTA
# (usadas por encoder.py para lookup rapido)
# ===========================================================================

def get_attack_item_mult(item_name: str, is_physical: bool, move_type, effectiveness: float) -> float:
    """
    Devuelve el multiplicador de daño del item del ATACANTE.
    Combina ATTACK_ITEMS y TYPE_BOOST_ITEMS.

    Args:
        item_name   : nombre del item (ya normalizado con clean())
        is_physical : True si el movimiento es fisico
        move_type   : PokemonType del movimiento
        effectiveness: multiplicador de efectividad (para Expert Belt)

    Returns:
        multiplicador total del item sobre el daño (default 1.0)
    """
    if item_name is None:
        return 1.0

    mult = 1.0

    # -- ATTACK_ITEMS --
    entry = ATTACK_ITEMS.get(item_name)
    if entry:
        item_mult = entry.get("mult", 1.0)
        category  = entry.get("category")
        supereff  = entry.get("supereffective_only", False)

        # filtro de categoria
        if category == "physical" and not is_physical:
            item_mult = 1.0
        elif category == "special" and is_physical:
            item_mult = 1.0

        # filtro de superefectividad (Expert Belt)
        if supereff and effectiveness < 2.0:
            item_mult = 1.0

        mult *= item_mult

    # -- TYPE_BOOST_ITEMS (x1.2 si el movimiento es del tipo correcto) --
    if move_type is not None and item_name in TYPE_BOOST_ITEMS:
        if TYPE_BOOST_ITEMS[item_name] == move_type:
            mult *= 1.2
        # Orbes de legendarios: tambien aplican al tipo secundario
        if item_name in LEGENDARY_ORB_SECONDARY:
            if LEGENDARY_ORB_SECONDARY[item_name] == move_type:
                mult *= 1.2

    return mult


def get_defense_item_divisor(item_name: str, is_physical: bool, move_type) -> float:
    """
    Devuelve el divisor de daño del item del DEFENSOR.
    divisor > 1.0 => reduce el daño (daño / divisor).

    Args:
        item_name   : nombre del item (ya normalizado)
        is_physical : True si el movimiento atacante es fisico
        move_type   : PokemonType del movimiento atacante

    Returns:
        divisor (default 1.0 = sin efecto)
    """
    if item_name is None:
        return 1.0

    entry = DEFENSE_ITEMS.get(item_name)
    if not entry:
        return 1.0

    divisor  = entry.get("divisor", 1.0)
    category = entry.get("category")
    itype    = entry.get("type")
    note     = entry.get("note", "")

    # Items reactivos (berries consumidas) — en la primera vez actuan
    # Si no podemos saber si ya se consumio, asumimos que esta disponible
    if "consumed" in note:
        # solo aplica si el movimiento es del tipo correcto
        if itype is not None and move_type != itype:
            return 1.0

    # filtro de categoria
    if category == "physical" and not is_physical:
        return 1.0
    if category == "special" and is_physical:
        return 1.0

    # filtro de tipo
    if itype is not None and isinstance(itype, PokemonType) and move_type != itype:
        return 1.0

    return divisor


def get_speed_item_mult(item_name: str) -> float:
    """
    Devuelve el multiplicador de velocidad del item.
    """
    if item_name is None:
        return 1.0
    entry = SPEED_ITEMS.get(item_name)
    if entry:
        return entry.get("mult", 1.0)
    return 1.0


def get_speed_ability_mult(ability_name: str, weather=None, field=None, has_status: bool = False,
                            item_lost: bool = False) -> float:
    """
    Devuelve el multiplicador de velocidad de la habilidad.
    Solo aplica si la condicion esta activa.

    Args:
        ability_name : nombre de la habilidad (ya normalizado)
        weather      : set de weathers activos (o None)
        field        : set de terrenos activos (o None)
        has_status   : True si el pokemon tiene un estado de problema
        item_lost    : True si el pokemon ha perdido su item (Unburden)
    """
    if ability_name is None:
        return 1.0

    entry = SPEED_ABILITIES.get(ability_name)
    if not entry:
        return 1.0

    mult      = entry.get("mult", 1.0)
    condition = entry.get("condition", "")

    from poke_env.battle import Weather, Field

    if condition == "rain":
        if weather and any(w in {Weather.RAINDANCE, Weather.PRIMORDIALSEA} for w in (weather or [])):
            return mult
        return 1.0
    elif condition == "sun":
        if weather and any(w in {Weather.SUNNYDAY, Weather.DESOLATELAND} for w in (weather or [])):
            return mult
        return 1.0
    elif condition == "sandstorm":
        if weather and Weather.SANDSTORM in (weather or []):
            return mult
        return 1.0
    elif condition == "hail_or_snow":
        if weather and any(w in {Weather.HAIL, Weather.SNOW} for w in (weather or [])):
            return mult
        return 1.0
    elif condition == "electric_terrain":
        if field and Field.ELECTRIC_TERRAIN in (field or []):
            return mult
        return 1.0
    elif condition == "status":
        return mult if has_status else 1.0
    elif condition == "item_lost":
        return mult if item_lost else 1.0
    elif condition == "first_5_turns":
        return mult  # simplificacion: siempre activo
    else:
        return 1.0  # condiciones no implementadas -> sin efecto


def get_attack_ability_params(ability_name: str) -> dict:
    """
    Devuelve el diccionario de parametros del atacante para su habilidad.
    Retorna {} si no existe.
    """
    return ATTACK_ABILITIES.get(ability_name, {})


def get_defense_ability_params(ability_name: str) -> dict:
    """
    Devuelve el diccionario de parametros del defensor para su habilidad.
    Retorna {} si no existe.
    """
    return DEFENSE_ABILITIES.get(ability_name, {})
