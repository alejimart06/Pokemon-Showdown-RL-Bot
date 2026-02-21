"""
reward.py
Funcion de reward para el agente RL.

El reward esta disenado para:
  1. Recompensar ganar y penalizar perder (reward terminal)
  2. Dar senales intermedias por:
     - Desmayar pokemon enemigos (+)
     - Perder pokemon propios (-)
     - Diferencia de HP acumulada al final

Las senales intermedias son importantes en Pokemon porque las partidas
pueden durar muchos turnos y sin ellas el agente tardaria mucho en aprender.
"""

from poke_env.environment import AbstractBattle


def compute_reward(
    last_battle: AbstractBattle,
    current_battle: AbstractBattle,
    config: dict,
) -> float:
    """
    Calcula el reward entre dos estados consecutivos de la batalla.

    Args:
        last_battle:    estado del turno anterior
        current_battle: estado actual
        config:         diccionario con los coeficientes de reward
                        (leido de config.yaml -> reward)

    Returns:
        float: reward del turno
    """
    # Coeficientes (con defaults si no estan en config)
    win_reward       = config.get("win", 1.0)
    lose_reward      = config.get("lose", -1.0)
    faint_enemy_r    = config.get("faint_enemy", 0.15)
    own_faint_r      = config.get("own_faint", -0.15)
    hp_coef          = config.get("hp_fraction_coef", 0.01)

    reward = 0.0

    # --- Reward terminal por ganar/perder ---
    if current_battle.won:
        reward += win_reward
    elif current_battle.lost:
        reward += lose_reward

    # --- Reward por desmayar pokemon enemigos ---
    prev_opp_fainted = _count_fainted(last_battle.opponent_team)
    curr_opp_fainted = _count_fainted(current_battle.opponent_team)
    new_opp_fainted  = curr_opp_fainted - prev_opp_fainted
    reward += new_opp_fainted * faint_enemy_r

    # --- Penalizacion por perder pokemon propios ---
    prev_own_fainted = _count_fainted(last_battle.team)
    curr_own_fainted = _count_fainted(current_battle.team)
    new_own_fainted  = curr_own_fainted - prev_own_fainted
    reward += new_own_fainted * own_faint_r

    # --- Reward por diferencia de HP al final de la partida ---
    # Solo al terminar para no dominar el reward intermedio
    if current_battle.finished:
        own_hp  = _total_hp_fraction(current_battle.team)
        opp_hp  = _total_hp_fraction(current_battle.opponent_team)
        reward += hp_coef * (own_hp - opp_hp)

    return reward


def _count_fainted(team: dict) -> int:
    """Cuenta cuantos pokemon del equipo han desmayado."""
    return sum(1 for p in team.values() if p.fainted)


def _total_hp_fraction(team: dict) -> float:
    """Suma de HP fractions de todos los pokemon del equipo."""
    return sum(p.current_hp_fraction for p in team.values())
