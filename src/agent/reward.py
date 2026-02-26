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
     - Penalizacion por cambiar innecesariamente (switch_penalty)
     - Recompensa por hacer setup / ganar boosts (boost_coef)
"""

from poke_env.battle import AbstractBattle


class RewardTracker:
    """
    Mantiene el estado anterior de la batalla para calcular
    deltas entre turnos (pokemon desmayados, switches, boosts, etc.).
    Se instancia por entorno y se resetea al inicio de cada batalla.
    """

    def __init__(self, config: dict):
        self.config = config
        self._prev_own_fainted  = 0
        self._prev_opp_fainted  = 0
        self._prev_active_name  = None   # para detectar switches
        self._prev_atk_boost    = 0      # boosts de ATK del turno anterior
        self._prev_spa_boost    = 0
        self._prev_spe_boost    = 0

    def reset(self):
        self._prev_own_fainted  = 0
        self._prev_opp_fainted  = 0
        self._prev_active_name  = None
        self._prev_atk_boost    = 0
        self._prev_spa_boost    = 0
        self._prev_spe_boost    = 0

    def compute(self, battle: AbstractBattle) -> float:
        config = self.config
        win_reward     = config.get("win", 1.0)
        lose_reward    = config.get("lose", -1.0)
        faint_enemy_r  = config.get("faint_enemy", 0.5)
        own_faint_r    = config.get("own_faint", -0.05)
        hp_coef        = config.get("hp_fraction_coef", 0.02)
        switch_penalty = config.get("switch_penalty", -0.02)
        boost_coef     = config.get("boost_coef", 0.03)

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

        # Switch penalty: penalizar si el activo cambio este turno
        own_active = battle.active_pokemon
        if own_active is not None:
            curr_name = getattr(own_active, 'species', None) or getattr(own_active, 'name', None)
            if self._prev_active_name is not None and curr_name != self._prev_active_name:
                reward += switch_penalty
            self._prev_active_name = curr_name

            # Boost reward: recompensar si se ganaron boosts ofensivos este turno
            curr_atk = own_active.boosts.get("atk", 0)
            curr_spa = own_active.boosts.get("spa", 0)
            curr_spe = own_active.boosts.get("spe", 0)

            delta = max(0, (curr_atk - self._prev_atk_boost)
                           + (curr_spa - self._prev_spa_boost)
                           + (curr_spe - self._prev_spe_boost))
            if delta > 0:
                reward += boost_coef * delta

            self._prev_atk_boost = curr_atk
            self._prev_spa_boost = curr_spa
            self._prev_spe_boost = curr_spe

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
    hp_coef       = config.get("hp_fraction_coef", 0.02)

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
