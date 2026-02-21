"""
reward.py
Funcion de reward para el agente RL.

En poke-env 0.11+, calc_reward solo recibe el estado actual de la batalla.
Para calcular senales intermedias (pokemon desmayados este turno) rastreamos
el estado anterior en el propio entorno.

El reward esta disenado para:
  1. Recompensar ganar y penalizar perder (reward terminal)
  2. Dar senales intermedias por:
     - Desmayar pokemon enemigos (+)
     - Perder pokemon propios (-)
     - Diferencia de HP acumulada al final
"""

from poke_env.battle import AbstractBattle


class RewardTracker:
    """
    Mantiene el estado anterior de la batalla para calcular
    deltas entre turnos (pokemon desmayados, etc.).
    Se instancia por entorno y se resetea al inicio de cada batalla.
    """

    def __init__(self, config: dict):
        self.config = config
        self._prev_own_fainted = 0
        self._prev_opp_fainted = 0

    def reset(self):
        self._prev_own_fainted = 0
        self._prev_opp_fainted = 0

    def compute(self, battle: AbstractBattle) -> float:
        config = self.config
        win_reward    = config.get("win", 1.0)
        lose_reward   = config.get("lose", -1.0)
        faint_enemy_r = config.get("faint_enemy", 0.15)
        own_faint_r   = config.get("own_faint", -0.15)
        hp_coef       = config.get("hp_fraction_coef", 0.01)

        reward = 0.0

        # Reward terminal
        if battle.won:
            reward += win_reward
        elif battle.lost:
            reward += lose_reward

        # Senales intermedias: deltas de pokemon desmayados
        curr_opp_fainted = _count_fainted(battle.opponent_team)
        curr_own_fainted = _count_fainted(battle.team)

        reward += (curr_opp_fainted - self._prev_opp_fainted) * faint_enemy_r
        reward += (curr_own_fainted - self._prev_own_fainted) * own_faint_r

        self._prev_opp_fainted = curr_opp_fainted
        self._prev_own_fainted = curr_own_fainted

        # Diferencia de HP al final de la partida
        if battle.finished:
            own_hp = _total_hp_fraction(battle.team)
            opp_hp = _total_hp_fraction(battle.opponent_team)
            reward += hp_coef * (own_hp - opp_hp)

        return reward


def compute_reward(battle: AbstractBattle, config: dict) -> float:
    """
    Calcula el reward para el estado actual de la batalla.
    Usado directamente cuando no se necesita tracking de estado previo.
    """
    win_reward    = config.get("win", 1.0)
    lose_reward   = config.get("lose", -1.0)
    hp_coef       = config.get("hp_fraction_coef", 0.01)

    reward = 0.0

    if battle.won:
        reward += win_reward
    elif battle.lost:
        reward += lose_reward

    if battle.finished:
        own_hp = _total_hp_fraction(battle.team)
        opp_hp = _total_hp_fraction(battle.opponent_team)
        reward += hp_coef * (own_hp - opp_hp)

    return reward


def _count_fainted(team: dict) -> int:
    return sum(1 for p in team.values() if p.fainted)


def _total_hp_fraction(team: dict) -> float:
    return sum(p.current_hp_fraction for p in team.values())
