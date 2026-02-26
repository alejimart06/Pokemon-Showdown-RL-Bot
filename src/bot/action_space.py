"""
action_space.py
Máscara de acciones válidas para MaskablePPO.

MAPEO DE ACCIONES (igual que SinglesEnv.action_to_order):
  -2:      default
  -1:      forfeit
  0-5:     switch a list(battle.team.values())[i]
  6-9:     movimiento (action-6) % 4
  10-13:   movimiento + mega
  14-17:   movimiento + z-move
  18-21:   movimiento + dynamax
  22-25:   movimiento + terastallize

CRÍTICO: los switches usan el ÍNDICE ABSOLUTO en battle.team.values(),
no un índice relativo a battle.available_switches. Por eso la máscara
debe recorrer todos los slots del equipo e iterar exactamente igual.
"""

from poke_env.battle import AbstractBattle
import numpy as np

ACTION_SPACE_SIZE = 26


def get_action_mask(battle: AbstractBattle) -> np.ndarray:
    """
    Devuelve una máscara bool[26] de acciones válidas para el turno actual.

    Sigue exactamente el mismo mapeo que SinglesEnv.action_to_order para
    garantizar que una acción marcada True siempre produce una orden legal.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    # Turno de espera (_wait=True): el servidor solo acepta /choose default.
    # available_moves/switches pueden no estar vacíos en este estado,
    # por eso hay que chequear _wait explícitamente.
    # Turno forzado: lista vacía o solo /choose default
    valid = battle.valid_orders
    if len(valid) == 0 or (len(valid) == 1 and str(valid[0]) == '/choose default'):
        mask[6] = True  # acción arbitraria; action_to_order lo convierte a default
        return mask

    # ---------------------------------------------------------------
    # Switches (acciones 0-5): slot i es válido si:
    #   - El pokemon en ese slot existe en el equipo
    #   - No está activo
    #   - No está fainted
    #   - Está en battle.available_switches
    # action_to_order usa list(battle.team.values())[action]
    # ---------------------------------------------------------------
    team = list(battle.team.values())
    available_switches_set = set(id(p) for p in battle.available_switches)

    for i, pokemon in enumerate(team):
        if i >= 6:
            break
        if id(pokemon) in available_switches_set:
            mask[i] = True

    # ---------------------------------------------------------------
    # Movimientos (acciones 6-9): slot i es válido si el pokemon activo
    # tiene al menos (i+1) movimientos conocidos y ese movimiento está
    # en battle.available_moves.
    # action_to_order usa active_pokemon.moves.values() indexado por (action-6)%4
    # ---------------------------------------------------------------
    active = battle.active_pokemon
    available_moves = battle.available_moves

    # Caso especial: struggle o recharge — solo 1 movimiento disponible
    if (len(available_moves) == 1 and
            available_moves[0].id in ("struggle", "recharge")):
        mask[6] = True  # solo acción 6 válida
    elif active is not None:
        move_list = list(active.moves.values())
        available_move_ids = set(m.id for m in available_moves)

        for i, move in enumerate(move_list):
            if i >= 4:
                break
            if move.id in available_move_ids:
                base = 6 + i
                mask[base] = True  # movimiento normal

                # Mega (10-13)
                if battle.can_mega_evolve:
                    mask[10 + i] = True

                # Z-move (14-17)
                if battle.can_z_move:
                    z_moves = battle.available_z_moves
                    z_ids = set(m.id for m in z_moves)
                    if move.id in z_ids:
                        mask[14 + i] = True

                # Dynamax (18-21)
                if battle.can_dynamax:
                    mask[18 + i] = True

                # Terastallize (22-25)
                if battle.can_tera:
                    mask[22 + i] = True

    # ---------------------------------------------------------------
    # Garantía mínima: si ninguna acción es válida es un turno forzado
    # (/choose default). Permitir solo acción 6 — action_to_order con
    # strict=False lo resolverá como default sin enviar orden inválida.
    # ---------------------------------------------------------------
    if not mask.any():
        mask[6] = True

    return mask
