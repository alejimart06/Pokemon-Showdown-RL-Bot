"""
action_space.py
Define y gestiona el espacio de acciones del agente.

En Random Battle el agente puede:
  - Usar uno de hasta 4 movimientos (indices 0-3)
  - Cambiar a uno de hasta 5 pokemon en reserva (indices 4-8)

Total: 9 acciones posibles (espacio discreto)

Algunas acciones pueden no estar disponibles en un turno concreto
(PP agotados, no hay pokemon en reserva, etc.). Las acciones no
disponibles se mascaran para que el agente no las elija.
"""

from poke_env.environment import AbstractBattle

N_MOVES = 4
N_SWITCHES = 5
ACTION_SPACE_SIZE = N_MOVES + N_SWITCHES  # 9


def action_to_move(action: int, battle: AbstractBattle):
    """
    Convierte un indice de accion en la orden concreta para poke-env.

    Devuelve:
      - Un objeto Move si la accion es un ataque
      - Un objeto Pokemon si la accion es un cambio
      - None si la accion no es valida en este turno
    """
    if action < N_MOVES:
        # Intentar usar movimiento
        moves = list(battle.available_moves)
        if action < len(moves):
            return moves[action]
        return None
    else:
        # Intentar cambiar de pokemon
        switch_index = action - N_MOVES
        switches = list(battle.available_switches)
        if switch_index < len(switches):
            return switches[switch_index]
        return None


def get_action_mask(battle: AbstractBattle) -> list[bool]:
    """
    Devuelve una lista de booleanos indicando que acciones estan
    disponibles en el turno actual.

    True  = accion disponible
    False = accion no disponible (el agente no debe elegirla)
    """
    mask = [False] * ACTION_SPACE_SIZE

    # Movimientos disponibles
    for i, _ in enumerate(battle.available_moves):
        if i < N_MOVES:
            mask[i] = True

    # Si no hay movimientos (struggle), el slot 0 se activa igualmente
    if not battle.available_moves:
        mask[0] = True

    # Cambios disponibles
    for i, _ in enumerate(battle.available_switches):
        if i < N_SWITCHES:
            mask[N_MOVES + i] = True

    return mask


def get_valid_action(action: int, battle: AbstractBattle) -> int:
    """
    Si la accion elegida por el agente no esta disponible,
    devuelve la primera accion valida como fallback.
    Esto evita errores en casos extremos.
    """
    mask = get_action_mask(battle)
    if mask[action]:
        return action

    # Fallback: primera accion disponible
    for i, available in enumerate(mask):
        if available:
            return i

    # No deberia llegar aqui
    return 0
