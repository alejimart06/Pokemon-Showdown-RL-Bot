"""
action_space.py
Documentacion del espacio de acciones gestionado por poke-env 0.11+.

En Gen9, SinglesEnv define automaticamente 26 acciones discretas:
  action = -2: default
  action = -1: forfeit
  0 <= action <= 5:   cambiar a uno de los 6 pokemon
  6 <= action <= 9:   usar movimiento 1-4
  10 <= action <= 13: usar movimiento + mega evolucionar
  14 <= action <= 17: usar movimiento + z-move
  18 <= action <= 21: usar movimiento + dynamax
  22 <= action <= 25: usar movimiento + terastallizar

poke-env gestiona internamente la conversion de indice -> BattleOrder
a traves de SinglesEnv.action_to_order(). No necesitamos reimplementarla.

Este modulo expone unicamente la constante ACTION_SPACE_SIZE para
que otros modulos puedan referenciarla si la necesitan.
"""

from poke_env.battle import AbstractBattle

# Tamanio del espacio de acciones para Gen9 (calculado por SinglesEnv)
# 6 switches + 4 moves * (1 + mega + z + dmax + tera) = 6 + 4*5 = 26
ACTION_SPACE_SIZE = 26


def get_action_mask(battle: AbstractBattle) -> list[bool]:
    """
    Devuelve una mascara de acciones validas para el turno actual.

    La mascara se usa con sb3-contrib MaskablePPO para que el agente
    no elija acciones ilegales (pokemon no disponible, sin PP, etc.).

    Mapeo de indices:
      0-5:   cambios a pokemon 0-5
      6-9:   movimientos 0-3
      10-13: movimientos 0-3 + mega
      14-17: movimientos 0-3 + z-move
      18-21: movimientos 0-3 + dynamax
      22-25: movimientos 0-3 + tera
    """
    mask = [False] * ACTION_SPACE_SIZE

    # Cambios disponibles (indices 0-5)
    for i, _ in enumerate(battle.available_switches):
        if i < 6:
            mask[i] = True

    moves = list(battle.available_moves)
    n_moves = len(moves)

    # Movimientos base (indices 6-9)
    for i in range(min(n_moves, 4)):
        mask[6 + i] = True

    # Si no hay movimientos disponibles, struggle siempre esta disponible
    if not moves:
        mask[6] = True

    # Mega (indices 10-13) - solo si puede mega evolucionar
    if battle.can_mega_evolve:
        for i in range(min(n_moves, 4)):
            mask[10 + i] = True

    # Z-move (indices 14-17) - solo si puede usar z-move
    if battle.can_z_move:
        available_z = battle.available_z_moves
        for i in range(min(len(available_z), 4)):
            mask[14 + i] = True

    # Dynamax (indices 18-21) - solo si puede dynamaxear
    if battle.can_dynamax:
        for i in range(min(n_moves, 4)):
            mask[18 + i] = True

    # Terastallizar (indices 22-25) - solo si puede terastallizar
    if battle.can_tera:
        for i in range(min(n_moves, 4)):
            mask[22 + i] = True

    return mask
